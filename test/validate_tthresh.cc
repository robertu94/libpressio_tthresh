#include <gtest/gtest.h>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <filesystem>
#include <cstdio>
#include <ios>
#include <libpressio_ext/cpp/libpressio.h>
#include <pressio_metrics.h>

struct input_description {
  std::string name;
  pressio_data metadata;
};

template <class T>
std::vector<T> read_to_vector(std::string const& filename) {
      std::vector<T> results;
      std::ifstream comp(filename);
      const auto begin = comp.tellg();
      comp.seekg(0, std::ios_base::end);
      const auto end = comp.tellg();
      comp.seekg(0, std::ios_base::beg);
      const size_t size_in_bytes = end-begin;
      const size_t elm_size = sizeof(T);
      const size_t nelems = size_in_bytes / elm_size;
      results.resize(nelems);
      comp.read(reinterpret_cast<char*>(results.data()), nelems);
      return results;
}

std::ostream& operator<< (std::ostream& out, input_description const& in) {
    return out <<
    '{' <<
     in.name << 
     ", " <<
     in.metadata << 
    '}'
    ;
}

class tthresh_test : public ::testing::TestWithParam<input_description> {
  public:
  void SetUp() {
    auto input_description = GetParam();
    filename = std::filesystem::path(CMAKE_PROJECT_SOURCE_DIR) / "data" / input_description.name;
    metadata = input_description.metadata;

    pressio library;
    auto metadata_clone = pressio_data::clone(metadata);
    pressio_io io = library.get_io("posix");
    io->set_options({
        {"io:path", filename}
        });
    pressio_data* tmp = io->read(&metadata_clone);
    if(tmp == nullptr) {
      FAIL() << "failed to read " << filename;
    }
    input_data = std::move(*tmp);
    pressio_data_free(tmp);
  }
  void TearDown() {
    cleanup();
  }

  protected:
    std::filesystem::path filename;
    pressio_data metadata;
    pressio_data input_data;

    void cleanup() const {
      //use error_code to prohibit exceptions if the file to delete does not exist
      std::error_code ec;
      std::filesystem::remove(compressed_filename(), ec);
      std::filesystem::remove(output_filename(), ec);
    }

    std::filesystem::path compressed_filename() const {
      return filename.string() + ".compressed_out";
    }
    std::filesystem::path output_filename() const {
      return filename.string() + ".decompressed_out";
    }

    std::string classic_type_str() const {
      switch (metadata.dtype()) {
        case pressio_uint8_dtype:
          return "uchar";
        case pressio_float_dtype:
          return "float";
        default:
          throw std::runtime_error("unsupported type");
      }
    }
    std::string options_to_classic_args_str(pressio_options const& options) const {
      std::string target_str;
      if(options.get("tthresh:target_str", &target_str) != pressio_options_key_set)  {
        throw std::runtime_error("required target_str not set");
      }
      double target_value;
      if(options.get("tthresh:target_value", &target_value) != pressio_options_key_set)  {
        throw std::runtime_error("required target_value not set");
      }

      std::stringstream ss;
      if (target_str == "eps") {
        ss << " -e ";
      } else if (target_str == "psnr") {
        ss << " -p ";
      } else if (target_str == "rel") {
        ss << " -r ";
      } else {
        throw std::runtime_error("unexpected target type");
      }
      ss << std::fixed << target_value;
      return ss.str();
    }

    std::string classic_dims_str() const {
      auto dims = metadata.dimensions(); 
      std::stringstream ss;
      for(auto it = dims.rbegin(); it != dims.rend(); ++it) {
        ss << *it << ' ';
      }

      return ss.str();
    }

    std::vector<uint8_t> run_tthresh_classic_compress(pressio_options const& options) const {

      std::stringstream cmd_ss;
      cmd_ss << 
        std::filesystem::path(TTHRESH_CLASSIC_CMD) << 
        " -v "
        " -i " << filename << 
        " -c " << compressed_filename() << 
        " -t " << classic_type_str() <<
        " -s " << classic_dims_str() <<
        options_to_classic_args_str(options)
        ;
      std::string cmd(cmd_ss.str());
      std::cout << cmd << std::endl;
      FILE* cmd_ex = popen(cmd.c_str(), "r");
      size_t nread;
      char buf[4096];
      while((nread = fread(buf, sizeof(char), sizeof(buf), cmd_ex)) > 0) {
        fwrite(buf, sizeof(char), nread, stdout);
      }
      pclose(cmd_ex);

      std::vector<uint8_t> results(read_to_vector<uint8_t>(compressed_filename()));
      return results;
    }

    std::vector<float> run_tthresh_classic_decompress() const {

      std::stringstream cmd_ss;
      cmd_ss << 
        std::filesystem::path(TTHRESH_CLASSIC_CMD) << 
        " -v " <<
        " -o " << output_filename() << 
        " -c " << compressed_filename()
        ;
      std::string cmd(cmd_ss.str());
      std::cout << cmd << std::endl;
      FILE* cmd_ex = popen(cmd.c_str(), "r");
      size_t nread;
      char buf[4096];
      while((nread = fread(buf, sizeof(char), sizeof(buf), cmd_ex)) > 0) {
        fwrite(buf, sizeof(char), nread, stdout);
      }
      pclose(cmd_ex);

      std::vector<float> results(read_to_vector<float>(compressed_filename()));
      return results;
    }

    std::vector<uint8_t> run_tthresh_libpressio_compress(pressio_options const& options) const {
      std::vector<uint8_t> results;

      pressio library;
      pressio_compressor compressor = library.get_compressor("tthresh");
      compressor->set_options(options);

      auto output_data = pressio_data::clone(metadata);

      compressor->compress(&input_data, &output_data);

      results.resize(output_data.size_in_bytes());
      std::memcpy(results.data(), output_data.data(), results.size());

      return results;
    }

    std::vector<float> run_tthresh_libpressio_decompress(std::vector<uint8_t>& input, pressio_options const& options) const {
      std::vector<float> results;

      pressio library;
      pressio_compressor compressor = library.get_compressor("tthresh");
      compressor->set_options(options);

      auto output_data = pressio_data::clone(metadata);
      pressio_data input_data = pressio_data::nonowning(
          pressio_byte_dtype, input.data() , {input.size()});

      compressor->decompress(&input_data, &output_data);

      results.resize(output_data.size_in_bytes());
      std::memcpy(results.data(), output_data.data(), results.size());

      return results;
    }
};


TEST_P(tthresh_test, classic_works) {
  double target_value = 1e-3;
  pressio_options options {
        {"tthresh:target_str", "eps"},
        {"tthresh:target_value", target_value},
  };
  auto classic_compressed = run_tthresh_classic_compress(options);
  auto classic_decompressed = run_tthresh_classic_decompress();

  pressio library;
  pressio_data metadata_clone = pressio_data::clone(metadata);
  pressio_io io = library.get_io("posix");
  io->set_options({
      {"io:path", output_filename()}
      });
  pressio_data* lp_decompressed_data = io->read(&metadata_clone);

  pressio_metrics error_stat = library.get_metric("error_stat");
  pressio_options* mr = pressio_metrics_evaluate(&error_stat, &input_data, nullptr, lp_decompressed_data);
  double max_eps = std::numeric_limits<double>::infinity();
  ASSERT_EQ(mr->get("error_stat:max_error", &max_eps), pressio_options_key_set) << *mr;
  ASSERT_LE(max_eps, target_value);
  pressio_options_free(mr);
}

TEST_P(tthresh_test, libpressio_works) {
  double target_value = 1e-3;
  pressio_options options {
        {"tthresh:target_str", "eps"},
        {"tthresh:target_value", target_value},
  };
  auto lp_compressed = run_tthresh_libpressio_compress(options);
  auto lp_decompressed = run_tthresh_libpressio_decompress(lp_compressed, options);

  auto lp_decompressed_data = pressio_data::nonowning(input_data.dtype(), lp_decompressed.data(), input_data.dimensions());

  pressio library;
  pressio_metrics error_stat = library.get_metric("error_stat");
  pressio_options* mr = pressio_metrics_evaluate(&error_stat, &input_data, nullptr, &lp_decompressed_data);
  double max_eps = std::numeric_limits<double>::infinity();
  ASSERT_EQ(mr->get("error_stat:max_error", &max_eps), pressio_options_key_set) << *mr;
  ASSERT_LE(max_eps, target_value);
  pressio_options_free(mr);

}


INSTANTIATE_TEST_SUITE_P(tthresh_validation,
    tthresh_test,
    testing::Values(
      input_description{"CLOUDf48.slice1.f32", pressio_data::empty(pressio_float_dtype, {500, 500, 10})})
    );
