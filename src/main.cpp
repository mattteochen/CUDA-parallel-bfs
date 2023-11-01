#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <map>
#include <utility>
#include <vector>

using std::cout;
using std::endl;

class Parser {
  std::map<uint32_t, std::vector<uint32_t>> m_adj;
  uint32_t m_num_nodes;
  std::string m_source_file;
  uint32_t m_total_neighbours;

public:
  Parser(std::string file) {
    m_source_file = file;
    m_num_nodes = 0;
    m_total_neighbours = 0;
  };

  uint32_t num_nodes() {
    return m_num_nodes;
  }

  uint32_t total_neighbours() {
    return m_total_neighbours;
  }

  auto& adj() const {
    return m_adj;
  }

  void print_adj() {
    for (auto& p : m_adj) {
      cout << "Node " << p.first << ": ";
      for (auto n : p.second) {
        cout << n << " ";
      }
      cout << endl;
    }
  }

  bool load_adj() {
    uint32_t lines = 0;
    constexpr char split_c = ' ';

    auto split = [](std::string& line) {
      std::string a = "", b = "";
      size_t i = 0;
      bool is_first = true;
      while (i < line.length()) {
        if (line[i] == split_c) {
          is_first = false;
          i++;
          continue;
        }
        if (is_first) {
          a += line[i];
        } else {
          b += line[i];
        }
        i++;
      }
      return std::make_pair(a, b);
    };

    std::ifstream file(m_source_file);
    if (file.is_open()) {
      std::string line;
      while (std::getline(file, line)) {
        if (!lines) {
          m_num_nodes = stoi(line);
          cout << "Total number of nodes: " << m_num_nodes << endl;
          lines++;
        } else {
          auto p = split(line);
          m_adj[stoi(p.first)].push_back(stoi(p.second));
        }
      }
      file.close();
    } else {
      std::cerr << "Unable to open the file." << endl;
      return 1;
    }
    for (auto& p : m_adj) {
      m_total_neighbours += p.second.size();
    }
    return 0;
  }
};

void fill_data(Parser& p, uint32_t num_nodes, uint32_t** node_ptr, uint32_t* node_neighbours) {
  auto& adj = p.adj();
  uint32_t index = 0;
  for (auto& [node, neighbours] : adj) {
    if (index < p.total_neighbours()) {
      node_ptr[node] = &(node_neighbours[index]);
    }
    for (auto n : neighbours) {
      node_neighbours[index++] = n;
    }
  }
  // for (uint32_t i=0; i<p.total_neighbours(); i++) {
  //   cout << node_neighbours[i] << " ";
  // }
  // cout << endl;
  // for (uint32_t i=0; i<num_nodes; i++) {
  //   if (node_ptr[i]) {
  //     cout << "node: " << i << " first neighbour: " << *(node_ptr[i]) << endl;
  //   } else {
  //     cout << "node " << i << " has no neighbours" << endl;
  //   }
  // }
  // cout << endl;
}

int main(int argc, char* argv[]) {
  Parser p("../tests/1.txt");
  if (p.load_adj()) {
    return 1;
  }
  p.print_adj();

  uint32_t num_nodes = p.num_nodes();
  uint32_t** node_ptr = (uint32_t**)malloc(num_nodes * sizeof(uint32_t*));
  for (uint32_t i=0; i<num_nodes; i++) {
    node_ptr[i] = 0;
  }
  uint32_t* node_neighbours = (uint32_t*)malloc(p.total_neighbours() * sizeof(uint32_t));
  for (uint32_t i=0; i<p.total_neighbours(); i++) {
    node_neighbours[i] = 0;
  }
  fill_data(p, num_nodes, node_ptr, node_neighbours);

  return 0;
}

