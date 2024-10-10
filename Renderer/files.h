#pragma once
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>

namespace nert_renderer {
    namespace fs = std::filesystem;

    inline void tokenize(std::string const& str, const char delim, std::vector<std::string>& out) {
        size_t start;
        size_t end = 0;

        while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
        {
            end = str.find(delim, start);
            out.push_back(str.substr(start, end - start));
        }
    }

    inline void removeQuote(std::vector<std::string>& strs) {
        for (auto& str : strs) {
            str.erase(std::remove(str.begin(), str.end(), '\"'), str.end());
        }
    }

    inline bool compareNat(const std::string& a, const std::string& b)
    {
        if (a.empty())
            return true;
        if (b.empty())
            return false;
        if (std::isdigit(a[0]) && !std::isdigit(b[0]))
            return true;
        if (!std::isdigit(a[0]) && std::isdigit(b[0]))
            return false;
        if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
        {
            if (std::toupper(a[0]) == std::toupper(b[0]))
                return compareNat(a.substr(1), b.substr(1));
            return (std::toupper(a[0]) < std::toupper(b[0]));
        }

        // Both strings begin with digit --> parse both numbers
        std::istringstream issa(a);
        std::istringstream issb(b);
        int ia, ib;
        issa >> ia;
        issb >> ib;
        if (ia != ib)
            return ia < ib;

        // Numbers are the same --> remove numbers and recurse
        std::string anew, bnew;
        std::getline(issa, anew);
        std::getline(issb, bnew);
        return (compareNat(anew, bnew));
    }


    inline void getFiles(std::vector<std::string>& directory, const fs::path& path) {
        if (fs::exists(path)) {
            for (auto& p : std::filesystem::directory_iterator(path))
                directory.push_back(p.path().string());
            std::sort(directory.begin(), directory.end(), compareNat);
        }
    }
}