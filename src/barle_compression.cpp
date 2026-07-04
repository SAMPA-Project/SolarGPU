#include "barle_compression.h"

#include <algorithm>

namespace solar_gpu::barle {

namespace {

struct Run { Byte width; int start; int len; };

// Classifies a delta value by the narrowest RunWidth that can hold it.
Byte determine_run_width(unsigned int x) {
    if (x <= 15) return kRunWidth4;
    if (x <= 255) return kRunWidth8;
    if (x <= 65535) return kRunWidth16;
    if (x <= 16777215) return kRunWidth24;
    return kRunWidth32;
}

void pack_int_bytes(unsigned int x, Byte width, std::vector<Byte>& out) {
    int nBytes = (width == kRunWidth4) ? 1 : (int)width;
    for (int i = 0; i < nBytes; ++i) {
        int shift = 8 * (nBytes - 1 - i);
        out.push_back((Byte)((x >> shift) & 0xFF));
    }
}

// Splits runs longer than 255 so each length fits in a single byte.
std::vector<Run> split_long_runs(const std::vector<Run>& runs) {
    std::vector<Run> out;
    out.reserve(runs.size());
    for (const Run& r : runs) {
        if (r.len <= 0) continue; // defensive: skip empty runs
        unsigned int remaining = (unsigned int)r.len;
        unsigned int start = (unsigned int)r.start;
        while (remaining > 255) {
            out.push_back({ r.width, (int)start, 255 });
            start += 255;
            remaining -= 255;
        }
        if (remaining > 0) out.push_back({ r.width, (int)start, (int)remaining });
    }
    return out;
}

/* Packs up to 8 run width codes (3 bits each) big-endian into exactly 3
   bytes. Unused slots are filled with kEndOfVoxel so the decoder knows
   where the block - and ultimately the per-voxel stream - ends. */
void pack_codes(const std::vector<Run>& runs, size_t base, std::vector<Byte>& out) {
    uint32_t bits = 0;
    for (int j = 0; j < 8; ++j) {
        Byte code = (base + (size_t)j < runs.size()) ? runs[base + j].width : kEndOfVoxel;
        bits |= (uint32_t)(code & 7) << ((7 - j) * 3);
    }
    out.push_back((Byte)((bits >> 16) & 0xFF));
    out.push_back((Byte)((bits >> 8) & 0xFF));
    out.push_back((Byte)(bits & 0xFF));
}

// One length byte per run in the block (lengths are <= 255 after splitting).
void pack_lengths(const std::vector<Run>& runs, size_t base, std::vector<Byte>& out) {
    size_t count = std::min<size_t>(runs.size() - base, 8);
    for (size_t j = 0; j < count; ++j) out.push_back((Byte)runs[base + j].len);
}

/* Packs the delta payload of every run in the block. 4-bit-width runs are
   packed two values per byte (high nibble first); wider widths are
   big-endian byte groups. */
void pack_deltas(const std::vector<unsigned int>& deltas, const std::vector<Run>& runs,
                  size_t base, std::vector<Byte>& out) {
    size_t count = std::min<size_t>(runs.size() - base, 8);
    for (size_t j = 0; j < count; ++j) {
        Byte width = runs[base + j].width;
        unsigned int pos = (unsigned int)runs[base + j].start;
        unsigned int len = (unsigned int)runs[base + j].len;
        for (unsigned int k = 0; k < len; ) {
            if (width == kRunWidth4) {
                unsigned int v1 = deltas[pos + k];
                unsigned int v2 = (k + 1 < len) ? deltas[pos + k + 1] : 0;
                out.push_back((Byte)((v1 << 4) | (v2 & 0x0F)));
                k += (k + 1 < len) ? 2 : 1;
            } else {
                pack_int_bytes(deltas[pos + k], width, out);
                ++k;
            }
        }
    }
}

std::vector<Run> detect_runs(const std::vector<unsigned int>& deltas) {
    std::vector<Run> runs;
    Byte prevWidth = determine_run_width(deltas[0]);
    Byte curWidth = prevWidth;
    int i = 0;
    while (i < (int)deltas.size()) {
        Run r{ prevWidth, i, 0 };
        int j = i + 1;
        while (j < (int)deltas.size() && (curWidth = determine_run_width(deltas[j])) == prevWidth) ++j;
        r.len = j - i;
        runs.push_back(r);
        i = j;
        prevWidth = curWidth;
    }
    return runs;
}

/* ba-RLE run optimization
   Pass 1: repeatedly absorb a single neighbouring element that is exactly
           one width class narrower into the wider run beside it (cheaper
           than paying a separate code+length for a single-element run).
   Pass 2: coalesce adjacent runs that ended up with the same width code. */
std::vector<Run> optimize_runs(const std::vector<Run>& input) {
    if (input.size() < 2) return input;

    std::vector<Run> current = input;
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<Run> next;
        int i = 0;
        while (i < (int)current.size() - 1) {
            if (current[i + 1].len == 1 && current[i + 1].width + 1 == current[i].width) {
                changed = true;
                Run merged = current[i];
                merged.len++;
                next.push_back(merged);
                i += 2;
            } else {
                next.push_back(current[i]);
                ++i;
            }
        }
        if (i < (int)current.size()) next.push_back(current[i]);

        if (next.size() == 1) return next;
        if (changed) current = next;
    }

    std::vector<Run> coalesced;
    int i = 0;
    while (i < (int)current.size()) {
        Run merged = current[i];
        int j = i + 1;
        while (j < (int)current.size() && merged.width == current[j].width) {
            merged.len += current[j].len;
            ++j;
        }
        coalesced.push_back(merged);
        i = j;
    }
    return coalesced;
}

} // namespace

std::vector<Byte> compress_voxel_neighbors(const int* raw, int max_neighbors, int& out_raw_bytes) {
    out_raw_bytes = 0;

    std::vector<unsigned int> ids;
    for (int k = 0; k < max_neighbors; ++k) {
        if (raw[k] == 0) break;
        ids.push_back((unsigned int)raw[k]);
    }
    if (ids.empty()) return {};

    std::sort(ids.begin(), ids.end(), std::greater<unsigned int>());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (!ids.empty() && ids.back() == 0) ids.pop_back();
    if (ids.empty()) return {};
    out_raw_bytes = (int)ids.size() * 4;

    std::vector<unsigned int> deltas(ids.size());
    deltas[0] = ids[0];
    for (size_t i = 1; i < ids.size(); ++i) deltas[i] = ids[i - 1] - ids[i];

    std::vector<Run> runs = detect_runs(deltas);
    std::vector<Run> optimized = optimize_runs(runs);
    std::vector<Run> expanded = split_long_runs(optimized);

    std::vector<Byte> encoded;
    for (size_t i = 0; i < expanded.size(); i += 8) {
        pack_codes(expanded, i, encoded);
        pack_lengths(expanded, i, encoded);
        pack_deltas(deltas, expanded, i, encoded);
    }
    return encoded;
}

void compress_all_neighbors(const int* raw_host, int num_total, int max_neighbors,
                             std::vector<int>& offsets, std::vector<Byte>& flat_data,
                             std::size_t& out_raw_bytes) {
    offsets.resize(num_total + 1);
    flat_data.clear();
    int offset = 0;
    out_raw_bytes = 0;

    for (int i = 0; i < num_total; ++i) {
        offsets[i] = offset;
        int rawBytes = 0;
        std::vector<Byte> compressed =
            compress_voxel_neighbors(raw_host + (size_t)i * max_neighbors, max_neighbors, rawBytes);
        out_raw_bytes += (size_t)rawBytes;
        flat_data.insert(flat_data.end(), compressed.begin(), compressed.end());
        offset += (int)compressed.size();
    }
    offsets[num_total] = offset;
}

} // namespace solar_gpu::barle
