#include "table_dist.h"

namespace nert_renderer {
    TableDist1D make_table_dist_1d(const std::vector<double>& f) {
        std::vector<double> pmf = f;
        std::vector<double> cdf(f.size() + 1);
        cdf[0] = 0;
        for (int i = 0; i < (int)f.size(); i++) {
            // assert(pmf[i] > 0);
            cdf[i + 1] = cdf[i] + pmf[i];
        }
        double total = cdf.back();
        if (total > 0) {
            for (int i = 0; i < (int)pmf.size(); i++) {
                pmf[i] /= total;
                cdf[i] /= total;
            }
        }
        else {
            for (int i = 0; i < (int)pmf.size(); i++) {
                pmf[i] = double(1) / double(pmf.size());
                cdf[i] = double(i) / double(pmf.size());
            }
            cdf.back() = 1;
        }

        return TableDist1D{ pmf, cdf };
    }

    int sample(const TableDist1D& table, double rnd_param) {
        int size = table.pmf.size();
        assert(size > 0);
        const double* ptr = std::upper_bound(table.cdf.data(), table.cdf.data() + size + 1, rnd_param);
        int offset = std::clamp(int(ptr - table.cdf.data() - 1), 0, size - 1);
        return offset;
    }

    double pmf(const TableDist1D& table, int id) {
        assert(id >= 0 && id < (int)table.pmf.size());
        return table.pmf[id];
    }

    TableDist2D make_table_dist_2d(const std::vector<double>& f, int width, int height) {
        // Construct a 1D distribution for each row
        std::vector<double> cdf_rows(height * (width + 1));
        std::vector<double> pdf_rows(height * width);
        for (int y = 0; y < height; y++) {
            cdf_rows[y * (width + 1)] = 0;
            for (int x = 0; x < width; x++) {
                cdf_rows[y * (width + 1) + (x + 1)] =
                    cdf_rows[y * (width + 1) + x] + f[y * width + x];
            }
            double integral = cdf_rows[y * (width + 1) + width];
            if (integral > 0) {
                // Normalize
                for (int x = 0; x < width; x++) {
                    cdf_rows[y * (width + 1) + x] /= integral;
                }
                // Setup the pmf/pdf
                for (int x = 0; x < width; x++) {
                    pdf_rows[y * width + x] = f[y * width + x] / integral;
                }
            }
            else {
                // don't care, since we do not care about it. 
                for (int x = 0; x < width; x++) {
                    pdf_rows[y * width + x] = 0;// double(1) / double(width);
                    cdf_rows[y * (width + 1) + x] = 0;// double(x) / double(width);
                }
                cdf_rows[y * (width + 1) + width] = 0;
            }
        }
        // Now construct the marginal CDF for each column.
        std::vector<double> cdf_marginals(height + 1);
        std::vector<double> pdf_marginals(height);
        cdf_marginals[0] = 0;
        for (int y = 0; y < height; y++) {
            double weight = cdf_rows[y * (width + 1) + width];
            cdf_marginals[y + 1] = cdf_marginals[y] + weight;
        }
        double total_values = cdf_marginals.back();
        if (total_values > 0) {
            // Normalize
            for (int y = 0; y < height; y++) {
                cdf_marginals[y] /= total_values;
            }
            cdf_marginals[height] = 1;
            // Setup pdf cols
            for (int y = 0; y < height; y++) {
                double weight = cdf_rows[y * (width + 1) + width];
                pdf_marginals[y] = weight / total_values;
            }
        }
        else {
            // The whole thing is black...why are we even here?
            // Still set up a uniform distribution.
            for (int y = 0; y < height; y++) {
                pdf_marginals[y] = double(1) / double(height);
                cdf_marginals[y] = double(y) / double(height);
            }
            cdf_marginals[height] = 1;
        }
        // We finally normalize the last entry of each cdf row to 1
        for (int y = 0; y < height; y++) {
            if (cdf_rows[y * (width + 1) + width - 1] > 0) {
                cdf_rows[y * (width + 1) + width] = 1;
            } else {
                cdf_rows[y * (width + 1) + width] = 0;
            }
        }

        return TableDist2D{
            cdf_rows, pdf_rows,
            cdf_marginals, pdf_marginals,
            total_values,
            width, height
        };
    }

    gdt::vec2f sample(const TableDist2D& table, const gdt::vec2f& rnd_param) {
        int w = table.width, h = table.height;
        // We first sample a row from the marginal distribution
        const double* y_ptr = std::upper_bound(
            table.cdf_marginals.data(),
            table.cdf_marginals.data() + h + 1,
            rnd_param[1]);
        int y_offset = std::clamp(int(y_ptr - table.cdf_marginals.data() - 1), 0, h - 1);
        // Uniformly remap rnd_param[1] to find the continuous offset 
        double dy = rnd_param[1] - table.cdf_marginals[y_offset];
        if ((table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]) > 0) {
            dy /= (table.cdf_marginals[y_offset + 1] - table.cdf_marginals[y_offset]);
        }
        // Sample a column at the row y_offset
        const double* cdf = &table.cdf_rows[y_offset * (w + 1)];
        const double* x_ptr = std::upper_bound(cdf, cdf + w + 1, rnd_param[0]);
        int x_offset = std::clamp(int(x_ptr - cdf - 1), 0, w - 1);
        // Uniformly remap rnd_param[0]
        double dx = rnd_param[0] - cdf[x_offset];
        if (cdf[x_offset + 1] - cdf[x_offset] > 0) {
            dx /= (cdf[x_offset + 1] - cdf[x_offset]);
        }
        return gdt::vec2f( (x_offset + dx) / w, (y_offset + dy) / h );
    }

    double pdf(const TableDist2D& table, const gdt::vec2f& xy) {
        // Convert xy to integer rows & columns
        int w = table.width, h = table.height;
        int x = std::clamp(double(xy.x * w), double(0), double(w - 1));
        int y = std::clamp(double(xy.y * h), double(0), double(h - 1));
        // What's the PDF for sampling row y?
        double pdf_y = table.pdf_marginals[y];
        // What's the PDF for sampling row x?
        double pdf_x = table.pdf_rows[y * w + x];
        return pdf_y * pdf_x * w * h;
    }
}