#pragma once
#include <vector>
#include "gdt/math/AffineSpace.h"

namespace nert_renderer {
    /// TableDist1D stores a tabular discrete distribution
    /// that we can sample from using the functions below.
    /// Useful for light source sampling.
    struct TableDist1D {
        std::vector<double> pmf;
        std::vector<double> cdf;
    };

    /// Construct the tabular discrete distribution given a vector of positive numbers.
    TableDist1D make_table_dist_1d(const std::vector<double>& f);

    /// Sample an entry from the discrete table given a random number in [0, 1]
    int sample(const TableDist1D& table, double rnd_param);

    /// The probability mass function of the sampling procedure above.
    double pmf(const TableDist1D& table, int id);

    /// TableDist2D stores a 2D piecewise constant distribution
    /// that we can sample from using the functions below.
    /// Useful for envmap sampling.
    struct TableDist2D {
        // cdf_rows & pdf_rows store a 1D piecewise constant distribution
        // for each row.
        std::vector<double> cdf_rows, pdf_rows;
        // cdf_maringlas & pdf_marginals store a single 1D piecewise
        // constant distribution for sampling a row
        std::vector<double> cdf_marginals, pdf_marginals;
        double total_values;
        int width, height;
    };

    /// Construct the 2D piecewise constant distribution given a vector of positive numbers
    /// and width & height.
    TableDist2D make_table_dist_2d(const std::vector<double>& f, int width, int height);

    /// Given two random number in [0, 1]^2, sample a point in the 2D domain [0, 1]^2
    /// with distribution proportional to f above.
    gdt::vec2f sample(const TableDist2D& table, const gdt::vec2f& rnd_param);

    /// Probability density of the sampling procedure above.
    double pdf(const TableDist2D& table, const gdt::vec2f& xy);
}
