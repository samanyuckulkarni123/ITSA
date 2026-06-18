#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

/*
ITSA cached C++ rewrite with DSSP-backed H/E/C assignment.

Goals:
- Preserve Python pipeline functionality as closely as possible
- Keep H/E/C secondary-structure logic via external mkdssp/dssp
- Add per-protein caching so tokens + self-scores are computed once
- Keep top-k interaction tokens, pi-pi geometry, overlap bonus, SW local alignment,
  and symmetric self-normalization

Compile:
  g++ -O3 -std=c++17 -pthread itsa_cached_dssp.cpp -o itsa_cached_dssp

Run example:
  ./itsa_cached_dssp --input_csv pairwise_dataset.csv --output_csv itsa_sw_results_cpp.csv \
      --base_dir SCOP_Dataset --workers 8 --dssp_bin mkdssp

Notes:
- This file intentionally avoids external C++ libraries for portability.
- It expects a DSSP-compatible executable (mkdssp or dssp) on PATH or via --dssp_bin.
- Exact score parity with Python should be very close if the same DSSP binary and same PDBs are used,
  though tiny floating-point differences are always possible.
*/

// ======================== BASIC MATH ========================

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator/(double d) const { return {x / d, y / d, z / d}; }
};

static inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static inline double norm(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

static inline double distance(const Vec3& a, const Vec3& b) {
    return norm(a - b);
}

static inline Vec3 normalize(const Vec3& v) {
    double n = norm(v);
    if (n == 0.0) return {0.0, 0.0, 0.0};
    return {v.x / n, v.y / n, v.z / n};
}

// ======================== UTIL ========================

static inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static inline std::string upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

static inline bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

static inline std::string join_path(const std::string& a, const std::string& b) {
    return (fs::path(a) / fs::path(b)).string();
}

static inline bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string format_seconds(double seconds) {
    long long s = std::max(0LL, static_cast<long long>(std::llround(seconds)));
    long long h = s / 3600;
    long long m = (s % 3600) / 60;
    long long sec = s % 60;
    std::ostringstream oss;
    if (h > 0) {
        oss << h << ':' << std::setw(2) << std::setfill('0') << m << ':' << std::setw(2) << sec;
    } else {
        oss << std::setw(2) << std::setfill('0') << m << ':' << std::setw(2) << sec;
    }
    return oss.str();
}

// ======================== DATA ========================

struct Atom {
    std::string name;
    char element = '\0';
    Vec3 coord;
};

struct Residue {
    std::string resname;
    char chain = 'A';
    int seq = 0;
    char icode = ' ';
    std::unordered_map<std::string, Atom> atoms;
};

struct ResidueToken {
    std::vector<char> interactions;
    char sec_struct = 'C';

    char primary() const {
        return interactions.empty() ? 'U' : interactions[0];
    }

    char to_letter() const {
        return primary();
    }
};

struct ProteinCacheEntry {
    bool ok = false;
    std::string status = "missing_file";
    std::vector<Residue> residues;
    std::vector<ResidueToken> tokens;
    double self_score = 0.0;
    int len = 0;
};

struct PrecomputedTokenRow {
    bool ok = false;
    std::string status = "missing_file";
    int token_length = 0;
    std::string pdb_id;
    std::string family;
    std::vector<ResidueToken> tokens;
    std::string error;
};

struct PairInputRow {
    std::string pdb_id_1;
    std::string family_1;
    std::string pdb_id_2;
    std::string family_2;
    std::string pair_label;
    std::string pair_type;
};

struct PairResultRow {
    std::string pdb_id_1;
    std::string family_1;
    std::string pdb_id_2;
    std::string family_2;
    std::string pair_label;
    std::string pair_type;
    double sw_raw_score = 0.0;
    double sw_norm = 0.0;
    double sw_self1_score = 0.0;
    double sw_self2_score = 0.0;
    int len1 = 0;
    int len2 = 0;
    std::string status = "missing_file";
};

struct ResidueKey {
    char chain = 'A';
    int seq = 0;
    char icode = ' ';

    bool operator==(const ResidueKey& o) const {
        return chain == o.chain && seq == o.seq && icode == o.icode;
    }
};

struct ResidueKeyHash {
    std::size_t operator()(const ResidueKey& k) const {
        std::size_t h1 = std::hash<int>{}(static_cast<int>(k.chain));
        std::size_t h2 = std::hash<int>{}(k.seq);
        std::size_t h3 = std::hash<int>{}(static_cast<int>(k.icode));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// ======================== CONFIG ========================

static constexpr double CANDIDATE_RESIDUE_SEARCH = 12.0;
static constexpr double HYDROGEN_BONDS_THRESH = 3.5;
static constexpr double DISULFIDE_THRESH = 2.3;
static constexpr double IONIC_THRESH = 4.0;
static constexpr double VDW_THRESH = 4.5;
static constexpr double PI_PI_THRESH = 5.5;
static constexpr double HYDROPHOBIC_THRESH = 4.8;
static constexpr double PI_PI_ANGLE_THRESH = 35.0;
static constexpr double DEFAULT_GAP_PENALTY = -2.0;
static constexpr int TOP_K = 3;

static const std::unordered_map<std::string, char> interaction_letters = {
    {"hydrogen_bonds", 'H'},
    {"disulfide_bridge", 'D'},
    {"ionic_bonds", 'I'},
    {"van_der_waals", 'V'},
    {"pi_pi_stacking", 'P'},
    {"hydrophobic_interaction", 'Y'}
};

static const std::unordered_set<std::string> hydrophobic_residues = {
    "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP", "GLY"
};
static const std::unordered_set<std::string> aromatic_residues = {
    "PHE", "TYR", "TRP", "HIS"
};
static const std::unordered_set<std::string> positive_residues = {
    "LYS", "ARG", "HIS"
};
static const std::unordered_set<std::string> negative_residues = {
    "ASP", "GLU"
};
static const std::unordered_set<std::string> backbone_atoms = {
    "N", "CA", "C", "O", "OXT"
};

static const std::unordered_map<std::string, std::unordered_set<std::string>> HBOND_ATOMS = {
    {"SER", {"OG"}},
    {"THR", {"OG1"}},
    {"ASN", {"OD1", "ND2"}},
    {"GLN", {"OE1", "NE2"}},
    {"TYR", {"OH"}},
    {"CYS", {"SG"}},
    {"HIS", {"ND1", "NE2"}},
    {"LYS", {"NZ"}},
    {"ARG", {"NE", "NH1", "NH2"}},
    {"ASP", {"OD1", "OD2"}},
    {"GLU", {"OE1", "OE2"}},
};

static const std::unordered_map<std::string, std::unordered_set<std::string>> IONIC_ATOMS = {
    {"LYS", {"NZ"}},
    {"ARG", {"NE", "NH1", "NH2"}},
    {"HIS", {"ND1", "NE2"}},
    {"ASP", {"OD1", "OD2"}},
    {"GLU", {"OE1", "OE2"}},
};

static const std::unordered_map<std::string, std::vector<std::string>> AROMATIC_RING_ATOMS = {
    {"PHE", {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}},
    {"TYR", {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"}},
    {"TRP", {"CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"}},
    {"HIS", {"CG", "ND1", "CD2", "CE1", "NE2"}},
};

static const std::unordered_map<std::string, std::unordered_set<std::string>> HYDROPHOBIC_ATOMS = {
    {"ALA", {"CB"}},
    {"VAL", {"CB", "CG1", "CG2"}},
    {"LEU", {"CB", "CG", "CD1", "CD2"}},
    {"ILE", {"CB", "CG1", "CG2", "CD1"}},
    {"MET", {"CB", "CG", "SD", "CE"}},
    {"PHE", {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}},
    {"PRO", {"CB", "CG", "CD"}},
    {"TRP", {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}},
    {"GLY", {"CA"}},
};

static const std::unordered_map<char, std::unordered_map<char, double>> interaction_substitution = {
    {'H', {{'H', 3}, {'I', 0}, {'Y', -1}, {'V', -1}, {'P', 0}, {'D', 0}, {'U', -1}}},
    {'I', {{'H', 0}, {'I', 3}, {'Y', -1}, {'V', -1}, {'P', 0}, {'D', 0}, {'U', -1}}},
    {'Y', {{'H', -1}, {'I', -1}, {'Y', 1}, {'V', 0}, {'P', -1}, {'D', -1}, {'U', -1}}},
    {'V', {{'H', -1}, {'I', -1}, {'Y', 0}, {'V', 0.5}, {'P', -1}, {'D', -1}, {'U', -1}}},
    {'P', {{'H', 0}, {'I', 0}, {'Y', -1}, {'V', -1}, {'P', 3}, {'D', 0}, {'U', -1}}},
    {'D', {{'H', 0}, {'I', 0}, {'Y', -1}, {'V', -1}, {'P', 0}, {'D', 3}, {'U', -1}}},
    {'U', {{'H', -1}, {'I', -1}, {'Y', -1}, {'V', -1}, {'P', -1}, {'D', -1}, {'U', 0}}},
};

// ======================== CSV ========================

static std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                cur.push_back('"');
                ++i;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

static std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n\r") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out += "\"";
    return out;
}

static std::vector<PairInputRow> read_input_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open input CSV: " + path);

    std::string header_line;
    if (!std::getline(in, header_line)) return {};
    auto headers = parse_csv_line(header_line);
    std::unordered_map<std::string, size_t> idx;
    for (size_t i = 0; i < headers.size(); ++i) idx[trim(headers[i])] = i;

    auto get = [&](const std::vector<std::string>& row, const std::string& key) -> std::string {
        auto it = idx.find(key);
        if (it == idx.end() || it->second >= row.size()) return "";
        return trim(row[it->second]);
    };

    std::vector<PairInputRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        auto row = parse_csv_line(line);
        PairInputRow r;
        r.pdb_id_1 = get(row, "pdb_id_1");
        r.family_1 = get(row, "family_1");
        r.pdb_id_2 = get(row, "pdb_id_2");
        r.family_2 = get(row, "family_2");
        r.pair_label = get(row, "pair_label");
        r.pair_type = get(row, "pair_type");
        rows.push_back(std::move(r));
    }
    return rows;
}

static void write_output_header(std::ofstream& out) {
    out << "pdb_id_1,family_1,pdb_id_2,family_2,pair_label,pair_type,"
        << "sw_raw_score,sw_norm,sw_self1_score,sw_self2_score,len1,len2,status\n";
}

static void write_result_row(std::ofstream& out, const PairResultRow& r) {
    out << csv_escape(r.pdb_id_1) << ','
        << csv_escape(r.family_1) << ','
        << csv_escape(r.pdb_id_2) << ','
        << csv_escape(r.family_2) << ','
        << csv_escape(r.pair_label) << ','
        << csv_escape(r.pair_type) << ','
        << std::fixed << std::setprecision(6) << r.sw_raw_score << ','
        << std::fixed << std::setprecision(6) << r.sw_norm << ','
        << std::fixed << std::setprecision(6) << r.sw_self1_score << ','
        << std::fixed << std::setprecision(6) << r.sw_self2_score << ','
        << r.len1 << ',' << r.len2 << ','
        << csv_escape(r.status) << '\n';
}


static std::string make_cache_key(const std::string& family, const std::string& pdb_id);
static double smith_waterman_tokens(const std::vector<ResidueToken>& seq1,
                                    const std::vector<ResidueToken>& seq2,
                                    double gap_penalty);

// ======================== PRECOMPUTED TOKEN CSV ========================

static std::string lower_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') return s.substr(1, s.size() - 2);
    return s;
}

static std::vector<char> parse_interactions_array(const std::string& text, size_t& pos) {
    std::vector<char> out;
    while (pos < text.size() && text[pos] != '[') ++pos;
    if (pos >= text.size()) return out;
    ++pos;

    while (pos < text.size()) {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
        if (pos < text.size() && text[pos] == ']') {
            ++pos;
            break;
        }
        if (pos < text.size() && text[pos] == '"') {
            ++pos;
            if (pos < text.size()) out.push_back(text[pos]);
            while (pos < text.size() && text[pos] != '"') ++pos;
            if (pos < text.size()) ++pos;
        } else {
            ++pos;
        }
    }
    return out;
}

static std::vector<ResidueToken> parse_tokens_json_minimal(const std::string& json_text) {
    std::vector<ResidueToken> tokens;
    std::string s = json_text;
    size_t pos = 0;

    while (true) {
        size_t inter_key = s.find("\"interactions\"", pos);
        if (inter_key == std::string::npos) break;
        pos = inter_key + 14;

        std::vector<char> interactions = parse_interactions_array(s, pos);
        if (interactions.empty()) interactions.push_back('U');

        char ss = 'C';
        size_t ss_key = s.find("\"sec_struct\"", pos);
        if (ss_key != std::string::npos) {
            size_t q1 = s.find('"', ss_key + 12);
            if (q1 != std::string::npos) {
                size_t q2 = s.find('"', q1 + 1);
                if (q2 != std::string::npos && q2 > q1 + 1) ss = s[q1 + 1];
                pos = q2 == std::string::npos ? pos : q2 + 1;
            }
        }

        tokens.push_back(ResidueToken{interactions, ss});
    }

    return tokens;
}

static std::unordered_map<std::string, PrecomputedTokenRow> read_precomputed_tokens_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Could not open tokens CSV: " + path);

    std::string header_line;
    if (!std::getline(in, header_line)) return {};
    auto headers = parse_csv_line(header_line);
    std::unordered_map<std::string, size_t> idx;
    for (size_t i = 0; i < headers.size(); ++i) idx[trim(headers[i])] = i;

    auto get = [&](const std::vector<std::string>& row, const std::string& key) -> std::string {
        auto it = idx.find(key);
        if (it == idx.end() || it->second >= row.size()) return "";
        return row[it->second];
    };

    std::unordered_map<std::string, PrecomputedTokenRow> out;
    std::string line;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        auto row = parse_csv_line(line);

        PrecomputedTokenRow r;
        r.pdb_id = trim(get(row, "pdb_id"));
        r.family = trim(get(row, "family"));
        r.status = trim(get(row, "status"));
        r.error = trim(get(row, "error"));
        try { r.token_length = std::stoi(trim(get(row, "token_length"))); } catch (...) { r.token_length = 0; }

        std::string tokens_json = get(row, "tokens_json");
        r.tokens = parse_tokens_json_minimal(tokens_json);
        if (r.status == "ok" && !r.tokens.empty()) r.ok = true;

        std::string key = make_cache_key(r.family, upper(r.pdb_id));
        out[key] = std::move(r);
    }
    return out;
}

static ProteinCacheEntry build_protein_cache_entry_from_precomputed(
    const std::unordered_map<std::string, PrecomputedTokenRow>& precomputed,
    const std::string& family,
    const std::string& pdb_id,
    double gap_penalty
) {
    ProteinCacheEntry entry;
    auto it = precomputed.find(make_cache_key(family, upper(pdb_id)));
    if (it == precomputed.end()) {
        entry.status = "missing_precomputed_token";
        return entry;
    }

    const auto& row = it->second;
    if (!row.ok || row.tokens.empty()) {
        entry.status = row.status.empty() ? "failed_seq" : row.status;
        return entry;
    }

    entry.tokens = row.tokens;
    entry.len = static_cast<int>(entry.tokens.size());
    entry.self_score = smith_waterman_tokens(entry.tokens, entry.tokens, gap_penalty);
    entry.status = "ok";
    entry.ok = true;
    return entry;
}

// ======================== PDB PARSING ========================

static char infer_element(const std::string& atom_name, const std::string& element_field) {
    std::string e = trim(element_field);
    if (!e.empty()) return static_cast<char>(std::toupper(static_cast<unsigned char>(e[0])));
    std::string a = trim(atom_name);
    if (a.empty()) return '\0';
    return static_cast<char>(std::toupper(static_cast<unsigned char>(a[0])));
}

static bool has_required_atoms(const Residue& residue) {
    return residue.atoms.count("N") && residue.atoms.count("CA") && residue.atoms.count("C");
}

static std::vector<Residue> parse_pdb_residues(const std::string& path) {
    std::ifstream in(path);
    std::vector<Residue> residues;
    if (!in) return residues;

    std::unordered_map<std::string, size_t> index;
    std::string line;
    while (std::getline(in, line)) {
        if (!(starts_with(line, "ATOM") || starts_with(line, "HETATM"))) continue;
        if (line.size() < 54) continue;

        std::string atom_name = trim(line.substr(12, 4));
        std::string resname = upper(trim(line.substr(17, 3)));
        char chain = (line.size() > 21 ? line[21] : 'A');
        int resseq = 0;
        try { resseq = std::stoi(trim(line.substr(22, 4))); } catch (...) { continue; }
        char icode = (line.size() > 26 ? line[26] : ' ');

        double x = 0.0, y = 0.0, z = 0.0;
        try {
            x = std::stod(line.substr(30, 8));
            y = std::stod(line.substr(38, 8));
            z = std::stod(line.substr(46, 8));
        } catch (...) {
            continue;
        }

        char elem = infer_element(atom_name, line.size() >= 78 ? line.substr(76, 2) : "");

        std::ostringstream key;
        key << chain << '|' << resseq << '|' << icode << '|' << resname;
        auto ks = key.str();

        if (!index.count(ks)) {
            index[ks] = residues.size();
            Residue r;
            r.resname = resname;
            r.chain = chain;
            r.seq = resseq;
            r.icode = icode;
            residues.push_back(std::move(r));
        }

        Atom atom;
        atom.name = atom_name;
        atom.element = elem;
        atom.coord = {x, y, z};
        residues[index[ks]].atoms[atom_name] = atom;
    }

    std::vector<Residue> filtered;
    filtered.reserve(residues.size());
    for (auto& r : residues) {
        if (has_required_atoms(r)) filtered.push_back(std::move(r));
    }
    return filtered;
}

// ======================== DSSP ========================

static char collapse_dssp_char(char ss_char) {
    if (ss_char == 'H' || ss_char == 'G' || ss_char == 'I') return 'H';
    if (ss_char == 'E' || ss_char == 'B') return 'E';
    return 'C';
}

static std::string temp_filename(const std::string& prefix, const std::string& suffix) {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::ostringstream oss;
    oss << prefix << '_' << std::this_thread::get_id() << '_' << now << suffix;
    return (fs::temp_directory_path() / oss.str()).string();
}

static std::unordered_map<ResidueKey, char, ResidueKeyHash>
assign_secondary_structure_dssp(const std::string& pdb_file, const std::string& dssp_bin) {
    std::unordered_map<ResidueKey, char, ResidueKeyHash> ss_map;

    std::string out_file = temp_filename("itsa_dssp", ".dssp");
    std::ostringstream cmd;
    cmd << dssp_bin
        << " --quiet --output-format=dssp "
        << '"' << pdb_file << '"'
        << " > "
        << '"' << out_file << '"'
        << " 2>/dev/null";
    int ret = std::system(cmd.str().c_str());
    if (ret != 0 || !file_exists(out_file)) {
        if (file_exists(out_file)) std::remove(out_file.c_str());
        return ss_map;
    }

    std::ifstream in(out_file);
    std::string line;
    bool data_started = false;
    while (std::getline(in, line)) {
        if (!data_started) {
            if (starts_with(line, "  #  RESIDUE")) data_started = true;
            continue;
        }
        if (line.size() < 17) continue;
        if (line[13] == '!') continue; // chain break or missing residue

        std::string seq_str = trim(line.substr(5, 5));
        if (seq_str.empty()) continue;

        int seq = 0;
        try { seq = std::stoi(seq_str); } catch (...) { continue; }
        char icode = (line.size() > 10 ? line[10] : ' ');
        char chain = (line.size() > 11 && line[11] != ' ' ? line[11] : 'A');
        char ss_raw = (line.size() > 16 ? line[16] : ' ');

        ResidueKey key{chain, seq, icode};
        ss_map[key] = collapse_dssp_char(ss_raw);
    }

    std::remove(out_file.c_str());
    return ss_map;
}

// ======================== INTERACTION HELPERS ========================

static std::vector<const Atom*> get_heavy_atoms(const Residue& residue,
                                                const std::unordered_set<std::string>* atom_names = nullptr,
                                                bool include_backbone = true) {
    std::vector<const Atom*> atoms;
    for (const auto& kv : residue.atoms) {
        const auto& name = kv.first;
        const auto& atom = kv.second;
        if (atom_names && !atom_names->count(name)) continue;
        if (!include_backbone && backbone_atoms.count(name)) continue;
        if (atom.element == 'H') continue;
        atoms.push_back(&atom);
    }
    return atoms;
}

static double min_atom_distance(const Residue& r1, const Residue& r2,
                                const std::unordered_set<std::string>* atom_names1 = nullptr,
                                const std::unordered_set<std::string>* atom_names2 = nullptr,
                                bool include_backbone1 = true,
                                bool include_backbone2 = true) {
    auto a1 = get_heavy_atoms(r1, atom_names1, include_backbone1);
    auto a2 = get_heavy_atoms(r2, atom_names2, include_backbone2);
    if (a1.empty() || a2.empty()) return std::numeric_limits<double>::infinity();

    double best = std::numeric_limits<double>::infinity();
    for (const Atom* p1 : a1) {
        for (const Atom* p2 : a2) {
            double d = distance(p1->coord, p2->coord);
            if (d < best) best = d;
        }
    }
    return best;
}

static std::optional<Vec3> get_aromatic_centroid(const Residue& residue) {
    auto it = AROMATIC_RING_ATOMS.find(residue.resname);
    if (it == AROMATIC_RING_ATOMS.end()) return std::nullopt;

    std::vector<Vec3> coords;
    for (const auto& atom_name : it->second) {
        auto a = residue.atoms.find(atom_name);
        if (a != residue.atoms.end()) coords.push_back(a->second.coord);
    }
    if (coords.size() < 3) return std::nullopt;

    Vec3 c{};
    for (const auto& p : coords) c = c + p;
    return c / static_cast<double>(coords.size());
}

static std::optional<Vec3> get_ring_plane_normal(const Residue& residue) {
    auto it = AROMATIC_RING_ATOMS.find(residue.resname);
    if (it == AROMATIC_RING_ATOMS.end()) return std::nullopt;

    std::vector<Vec3> coords;
    for (const auto& atom_name : it->second) {
        auto a = residue.atoms.find(atom_name);
        if (a != residue.atoms.end()) coords.push_back(a->second.coord);
    }
    if (coords.size() < 3) return std::nullopt;

    Vec3 p1 = coords[0], p2 = coords[1], p3 = coords[2];
    Vec3 v1 = p2 - p1;
    Vec3 v2 = p3 - p1;
    Vec3 n = cross(v1, v2);
    if (norm(n) == 0.0) return std::nullopt;
    return normalize(n);
}

static double angle_between_normals(const std::optional<Vec3>& n1, const std::optional<Vec3>& n2) {
    if (!n1 || !n2) return 180.0;
    double d = std::clamp(std::abs(dot(*n1, *n2)), -1.0, 1.0);
    return std::acos(d) * 180.0 / M_PI;
}

static bool has_hbond_capability(const Residue& residue) {
    return HBOND_ATOMS.count(residue.resname) > 0;
}

static bool is_opposite_charge_pair(const Residue& r1, const Residue& r2) {
    bool p1 = positive_residues.count(r1.resname) > 0;
    bool p2 = positive_residues.count(r2.resname) > 0;
    bool n1 = negative_residues.count(r1.resname) > 0;
    bool n2 = negative_residues.count(r2.resname) > 0;
    return (p1 && n2) || (n1 && p2);
}

static std::pair<std::string, double> classify_interaction(const Residue& r1, const Residue& r2) {
    if (r1.resname == "CYS" && r2.resname == "CYS" && r1.atoms.count("SG") && r2.atoms.count("SG")) {
        double sg_distance = distance(r1.atoms.at("SG").coord, r2.atoms.at("SG").coord);
        if (sg_distance <= DISULFIDE_THRESH) return {"disulfide_bridge", sg_distance};
    }

    if (is_opposite_charge_pair(r1, r2)) {
        auto it1 = IONIC_ATOMS.find(r1.resname);
        auto it2 = IONIC_ATOMS.find(r2.resname);
        double ionic_distance = min_atom_distance(
            r1, r2,
            it1 == IONIC_ATOMS.end() ? nullptr : &it1->second,
            it2 == IONIC_ATOMS.end() ? nullptr : &it2->second,
            true, true
        );
        if (ionic_distance <= IONIC_THRESH) return {"ionic_bonds", ionic_distance};
    }

    if (has_hbond_capability(r1) && has_hbond_capability(r2)) {
        auto it1 = HBOND_ATOMS.find(r1.resname);
        auto it2 = HBOND_ATOMS.find(r2.resname);
        double hbond_distance = min_atom_distance(
            r1, r2,
            &it1->second,
            &it2->second,
            true, true
        );
        if (hbond_distance <= HYDROGEN_BONDS_THRESH) return {"hydrogen_bonds", hbond_distance};
    }

    if (aromatic_residues.count(r1.resname) && aromatic_residues.count(r2.resname)) {
        auto centroid1 = get_aromatic_centroid(r1);
        auto centroid2 = get_aromatic_centroid(r2);
        if (centroid1 && centroid2) {
            double centroid_distance = distance(*centroid1, *centroid2);
            double ring_angle = angle_between_normals(
                get_ring_plane_normal(r1),
                get_ring_plane_normal(r2)
            );
            if (centroid_distance <= PI_PI_THRESH && ring_angle <= PI_PI_ANGLE_THRESH) {
                return {"pi_pi_stacking", centroid_distance};
            }
        }
    }

    if (hydrophobic_residues.count(r1.resname) && hydrophobic_residues.count(r2.resname)) {
        auto it1 = HYDROPHOBIC_ATOMS.find(r1.resname);
        auto it2 = HYDROPHOBIC_ATOMS.find(r2.resname);
        double hydrophobic_distance = min_atom_distance(
            r1, r2,
            it1 == HYDROPHOBIC_ATOMS.end() ? nullptr : &it1->second,
            it2 == HYDROPHOBIC_ATOMS.end() ? nullptr : &it2->second,
            false, false
        );
        if (hydrophobic_distance <= HYDROPHOBIC_THRESH) {
            return {"hydrophobic_interaction", hydrophobic_distance};
        }
    }

    double vdw_distance = min_atom_distance(r1, r2, nullptr, nullptr, false, false);
    if (!std::isfinite(vdw_distance)) vdw_distance = min_atom_distance(r1, r2);
    if (vdw_distance <= VDW_THRESH) return {"van_der_waals", vdw_distance};

    return {"", std::numeric_limits<double>::infinity()};
}

// ======================== TOKEN EXTRACTION ========================

static std::vector<ResidueToken> extract_interaction_tokens(
    const std::string& pdb_file,
    const std::string& dssp_bin,
    std::vector<Residue>* out_residues = nullptr
) {
    std::vector<Residue> residues = parse_pdb_residues(pdb_file);
    if (residues.empty()) {
        if (out_residues) *out_residues = {};
        return {};
    }

    auto ss_map = assign_secondary_structure_dssp(pdb_file, dssp_bin);

    std::vector<ResidueToken> tokens;
    tokens.reserve(residues.size());

    for (size_t i = 0; i < residues.size(); ++i) {
        const Residue& res1 = residues[i];
        auto ca_it = res1.atoms.find("CA");
        if (ca_it == res1.atoms.end()) continue;
        const Vec3& ca1 = ca_it->second.coord;

        std::vector<std::pair<std::string, double>> candidate_interactions;
        candidate_interactions.reserve(32);

        for (size_t j = 0; j < residues.size(); ++j) {
            if (i == j) continue;
            const Residue& res2 = residues[j];
            auto ca2_it = res2.atoms.find("CA");
            if (ca2_it == res2.atoms.end()) continue;
            if (distance(ca1, ca2_it->second.coord) > CANDIDATE_RESIDUE_SEARCH) continue;

            auto [interaction_type, interaction_distance] = classify_interaction(res1, res2);
            if (!interaction_type.empty()) {
                candidate_interactions.push_back({interaction_type, interaction_distance});
            }
        }

        std::sort(candidate_interactions.begin(), candidate_interactions.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second < b.second;
            return a.first < b.first;
        });

        std::unordered_set<std::string> seen_types;
        std::vector<char> letters;
        for (const auto& [itype, dist] : candidate_interactions) {
            (void)dist;
            if (!seen_types.insert(itype).second) continue;
            letters.push_back(interaction_letters.at(itype));
            if (static_cast<int>(letters.size()) >= TOP_K) break;
        }
        if (letters.empty()) letters.push_back('U');

        ResidueKey key{res1.chain, res1.seq, res1.icode};
        char ss = 'C';
        auto it = ss_map.find(key);
        if (it != ss_map.end()) ss = it->second;

        tokens.push_back(ResidueToken{letters, ss});
    }

    if (out_residues) *out_residues = std::move(residues);
    return tokens;
}

// ======================== SCORING ========================

static double score_tokens(const ResidueToken& t1, const ResidueToken& t2) {
    char p1 = t1.primary();
    char p2 = t2.primary();

    double s = -1.0;
    auto row_it = interaction_substitution.find(p1);
    if (row_it != interaction_substitution.end()) {
        auto col_it = row_it->second.find(p2);
        if (col_it != row_it->second.end()) s = col_it->second;
    }

    if (t1.sec_struct == t2.sec_struct) {
        if (t1.sec_struct == 'H') s += 0.2;
        else if (t1.sec_struct == 'E') s += 0.8;
    } else {
        bool s1 = (t1.sec_struct == 'H' || t1.sec_struct == 'E');
        bool s2 = (t2.sec_struct == 'H' || t2.sec_struct == 'E');
        if (s1 && s2) s -= 0.3;
    }

    std::unordered_set<char> extra1, extra2;
    for (size_t i = 1; i < t1.interactions.size(); ++i) extra1.insert(t1.interactions[i]);
    for (size_t i = 1; i < t2.interactions.size(); ++i) extra2.insert(t2.interactions[i]);

    int overlap = 0;
    for (char c : extra1) if (extra2.count(c)) ++overlap;
    s += 0.35 * overlap;

    return s;
}

static double smith_waterman_tokens(const std::vector<ResidueToken>& seq1,
                                    const std::vector<ResidueToken>& seq2,
                                    double gap_penalty) {
    const size_t n = seq1.size();
    const size_t m = seq2.size();
    if (n == 0 || m == 0) return 0.0;

    std::vector<double> prev(m + 1, 0.0), curr(m + 1, 0.0);
    double best_score = 0.0;

    for (size_t i = 1; i <= n; ++i) {
        curr[0] = 0.0;
        for (size_t j = 1; j <= m; ++j) {
            double diag = prev[j - 1] + score_tokens(seq1[i - 1], seq2[j - 1]);
            double up = prev[j] + gap_penalty;
            double left = curr[j - 1] + gap_penalty;
            curr[j] = std::max({0.0, diag, up, left});
            if (curr[j] > best_score) best_score = curr[j];
        }
        std::swap(prev, curr);
    }

    return best_score;
}

// ======================== CACHE ========================

static std::string make_cache_key(const std::string& family, const std::string& pdb_id) {
    return family + "||" + pdb_id;
}

static std::string make_pdb_path(const std::string& base_dir, const std::string& family, const std::string& pdb_id) {
    std::string pdb_lower = pdb_id;
    for (char& c : pdb_lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return join_path(join_path(base_dir, family), pdb_lower + ".pdb");
}

static ProteinCacheEntry build_protein_cache_entry(
    const std::string& pdb_file,
    const std::string& dssp_bin,
    double gap_penalty
) {
    ProteinCacheEntry entry;
    if (!file_exists(pdb_file)) {
        entry.status = "missing_file";
        return entry;
    }

    try {
        entry.tokens = extract_interaction_tokens(pdb_file, dssp_bin, &entry.residues);
        entry.len = static_cast<int>(entry.tokens.size());
        if (entry.tokens.empty()) {
            entry.status = "failed_seq";
            entry.ok = false;
            return entry;
        }
        entry.self_score = smith_waterman_tokens(entry.tokens, entry.tokens, gap_penalty);
        entry.status = "ok";
        entry.ok = true;
        return entry;
    } catch (const std::exception& e) {
        entry.status = std::string("error: ") + e.what();
        entry.ok = false;
        return entry;
    } catch (...) {
        entry.status = "error: unknown";
        entry.ok = false;
        return entry;
    }
}

// ======================== PAIRWISE ========================

static PairResultRow compare_cached_pair(
    const PairInputRow& row,
    const ProteinCacheEntry& e1,
    const ProteinCacheEntry& e2,
    double gap_penalty
) {
    PairResultRow out;
    out.pdb_id_1 = row.pdb_id_1;
    out.family_1 = row.family_1;
    out.pdb_id_2 = row.pdb_id_2;
    out.family_2 = row.family_2;
    out.pair_label = row.pair_label;
    out.pair_type = row.pair_type;

    out.len1 = e1.len;
    out.len2 = e2.len;
    out.sw_self1_score = e1.self_score;
    out.sw_self2_score = e2.self_score;

    if (!e1.ok) {
        out.status = (e1.status == "missing_file") ? "missing_file" : "failed_seq1";
        return out;
    }
    if (!e2.ok) {
        out.status = (e2.status == "missing_file") ? "missing_file" : "failed_seq2";
        return out;
    }

    double sw_score_12 = smith_waterman_tokens(e1.tokens, e2.tokens, gap_penalty);
    double sw_norm = 0.0;
    if (e1.self_score > 0.0 && e2.self_score > 0.0) {
        sw_norm = 2.0 * sw_score_12 / (e1.self_score + e2.self_score);
    }
    sw_norm = std::max(0.0, std::min(1.0, sw_norm));

    out.sw_raw_score = sw_score_12;
    out.sw_norm = sw_norm * 100.0;
    out.status = "ok";
    return out;
}

// ======================== BATCH ========================

struct Args {
    std::string input_csv;
    std::string output_csv = "itsa_sw_results_cpp.csv";
    std::string base_dir = "SCOP_Dataset";
    std::string dssp_bin = "mkdssp";
    std::string tokens_csv = "protein_tokens.csv";
    double gap_penalty = DEFAULT_GAP_PENALTY;
    int workers = 1;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
            return argv[++i];
        };

        if (a == "--input_csv") args.input_csv = next(a);
        else if (a == "--output_csv") args.output_csv = next(a);
        else if (a == "--base_dir") args.base_dir = next(a);
        else if (a == "--dssp_bin") args.dssp_bin = next(a);
        else if (a == "--gap") args.gap_penalty = std::stod(next(a));
        else if (a == "--tokens_csv") args.tokens_csv = next(a);
        else if (a == "--workers") args.workers = std::max(1, std::stoi(next(a)));
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: ./itsa_cached_dssp --input_csv file.csv [--output_csv out.csv] [--tokens_csv protein_tokens.csv] [--base_dir dir] [--dssp_bin mkdssp] [--gap -2.0] [--workers 8]\n";
            std::exit(0);
        }
    }
    if (args.input_csv.empty()) throw std::runtime_error("Provide --input_csv");
    return args;
}

static void run_batch(const Args& args) {
    auto rows = read_input_csv(args.input_csv);
    const size_t total = rows.size();
    if (total == 0) throw std::runtime_error("No rows found in input CSV.");

    std::unordered_map<std::string, std::pair<std::string, std::string>> unique_proteins;
    unique_proteins.reserve(total * 2);
    for (const auto& row : rows) {
        unique_proteins[make_cache_key(row.family_1, row.pdb_id_1)] = {row.family_1, row.pdb_id_1};
        unique_proteins[make_cache_key(row.family_2, row.pdb_id_2)] = {row.family_2, row.pdb_id_2};
    }

    auto precomputed = read_precomputed_tokens_csv(args.tokens_csv);
    std::cout << "Loaded precomputed token rows: " << precomputed.size() << "\n";

    std::unordered_map<std::string, ProteinCacheEntry> cache;
    cache.reserve(unique_proteins.size());

    std::vector<std::pair<std::string, std::pair<std::string, std::string>>> protein_list;
    protein_list.reserve(unique_proteins.size());
    for (const auto& kv : unique_proteins) protein_list.push_back(kv);
    std::sort(protein_list.begin(), protein_list.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::cout << "Unique proteins to cache: " << protein_list.size() << "\n";

    auto cache_start = std::chrono::steady_clock::now();
    if (args.workers <= 1) {
        size_t done = 0;
        for (const auto& item : protein_list) {
            const auto& key = item.first;
            const auto& family = item.second.first;
            const auto& pdb_id = item.second.second;
            cache[key] = build_protein_cache_entry_from_precomputed(precomputed, family, pdb_id, args.gap_penalty);
            ++done;
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - cache_start).count();
            double avg = elapsed / done;
            double eta = avg * (protein_list.size() - done);
            std::cout << "cache " << done << "/" << protein_list.size() << " | elapsed " << format_seconds(elapsed) << " | ETA " << format_seconds(eta) << "\n";
        }
    } else {
        std::vector<std::future<std::pair<std::string, ProteinCacheEntry>>> futs;
        futs.reserve(protein_list.size());
        for (const auto& item : protein_list) {
            futs.push_back(std::async(std::launch::async, [&, item]() {
                const auto& key = item.first;
                const auto& family = item.second.first;
                const auto& pdb_id = item.second.second;
                return std::make_pair(key, build_protein_cache_entry_from_precomputed(precomputed, family, pdb_id, args.gap_penalty));
            }));
        }

        size_t done = 0;
        for (auto& fut : futs) {
            auto res = fut.get();
            cache[res.first] = std::move(res.second);
            ++done;
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - cache_start).count();
            double avg = elapsed / done;
            double eta = avg * (protein_list.size() - done);
            std::cout << "cache " << done << "/" << protein_list.size() << " | elapsed " << format_seconds(elapsed) << " | ETA " << format_seconds(eta) << "\n";
        }
    }

    std::ofstream out(args.output_csv);
    if (!out) throw std::runtime_error("Could not open output CSV: " + args.output_csv);
    write_output_header(out);

    auto pair_start = std::chrono::steady_clock::now();
    size_t completed = 0;
    for (const auto& row : rows) {
        const auto& e1 = cache.at(make_cache_key(row.family_1, row.pdb_id_1));
        const auto& e2 = cache.at(make_cache_key(row.family_2, row.pdb_id_2));
        auto result = compare_cached_pair(row, e1, e2, args.gap_penalty);
        write_result_row(out, result);
        ++completed;

        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - pair_start).count();
        double avg = elapsed / completed;
        double eta = avg * (total - completed);
        std::cout << completed << "/" << total << " done | elapsed " << format_seconds(elapsed) << " | ETA " << format_seconds(eta) << "\n";
    }

    std::cout << "Saved batch results to " << args.output_csv << "\n";
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        run_batch(args);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unknown error\n";
        return 1;
    }
}