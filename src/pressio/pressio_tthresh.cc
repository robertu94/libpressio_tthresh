#include <libpressio_ext/cpp/compressor.h>
#include <libpressio_ext/cpp/pressio.h>
#include <libpressio_ext/cpp/domain_manager.h>
#include <std_compat/memory.h>
#include <map>
#include "tthresh_lib.h"
#include "tthresh_metrics.h"

extern "C" void libpressio_register_tthresh() {
  
}

const std::map<std::string, Target> target_types {
  {"eps", Target::eps},
  {"rmse", Target::rmse},
  {"psnr", Target::psnr},
};
std::vector<std::string> target_types_strs() {
  std::vector<std::string> s;
  std::transform(target_types.begin(), target_types.end(), std::back_inserter(s),
      [](decltype(target_types)::const_reference it){
       return it.first;
      });
  return s;
}

class tthresh_compressor_plugin: public libpressio_compressor_plugin {
public:
  std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<tthresh_compressor_plugin>(*this);
  }
  const char *prefix() const override {
    return "tthresh";
  }
  const char *version() const override {
    return "3c876c06d60570cc915bb61b43d2094ff1847e42";
  }

  int patch_version() const override {
    return 2;
  }

protected:
  pressio_options get_options_impl() const override {
    pressio_options opts;
    if(target == Target::eps) {
      set(opts, "pressio:abs", target_value);
    } else {
      set_type(opts, "pressio:abs", pressio_option_double_type);
    }
    set_type(opts, "tthresh:target_str", pressio_option_charptr_type);
    set(opts, "tthresh:target", static_cast<int32_t>(target));
    set(opts, "tthresh:target_value", target_value);
    return opts;
  }
  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set(opts, "pressio:description", R"(TTHRESH: Tensor Compression for Multidimensional Visual Data

    It is intended for Cartesian grid data of **3 or more dimensions**, and leverages the higher-order singular value decomposition (HOSVD), a generalization of the SVD to 3 and more dimensions.
    See also [TTHRESH: Tensor Compression for Multidimensional Visual Data (R. Ballester-Ripoll, P. Lindstrom and R. Pajarola)](https://arxiv.org/pdf/1806.05952.pdf)
    )");
    set(opts, "tthresh:target_str", "string that represents the target type");
    set(opts, "tthresh:target", "numeric code for the target type");
    set(opts, "tthresh:target_value", "the value of the target");
    return opts;
  }
  pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts,"pressio:thread_safe", pressio_thread_safety_single);
    set(opts,"pressio:stability", "experimental");
    set(opts,"tthresh:target_str", target_types_strs());
    

    
        std::vector<std::string> invalidations {"tthresh:target_value", "pressio:abs", "tthresh:target_str", "tthresh:target"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(opts, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(opts, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(opts, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
        set(opts, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:abs"}));

    return opts;
  }
  int set_options_impl(const pressio_options &options) override {
    if(get(options, "pressio:abs", &target_value) == pressio_options_key_set) {
      target = eps;
    }
    get(options, "tthresh:target_value", &target_value);
    int32_t tmp_target;
    std::string tmp_target_str;
    if(get(options, "tthresh:target_str", &tmp_target_str) == pressio_options_key_set) {
      auto it = target_types.find(tmp_target_str);
      if(it != target_types.end()) {
        target = it->second;
      } else {
        return set_error(1, "invalid target type: " + tmp_target_str);
      }
    } else if(get(options, "tthresh:target", &tmp_target) == pressio_options_key_set) {
      Target target_typed = static_cast<Target>(tmp_target);
      if(target_typed == Target::eps || target_typed == Target::psnr || target_typed == Target::rmse) {
        target = target_typed;
      } else {
        return set_error(1, "invalid target type");
      }
    }

    return 0;
  }
  int compress_impl(const pressio_data *real_input, struct pressio_data *output) override {
    pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    auto dims = input.dimensions();
    s = std::vector<uint32_t>(dims.begin(), dims.end());
    cumulative_products(s, sprod);
    try {
    compress_pressio_tthresh(
        &input,
        output,
        target,
        target_value,
        metrics
        );
    return 0;
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }
  int decompress_impl(const pressio_data *real_input, struct pressio_data *output) override {
    pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    std::vector<Slice> slices;
    try {
    decompress_pressio_tthresh(
        &input,
        output,
        slices,
        false,
        metrics
        );
      return 0;
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }
  pressio_options get_metrics_results_impl() const override {
    pressio_options ret;
    set(ret, "tthresh:encoded_core_size", metrics.encoded_core_size);
    set(ret, "tthresh:unweighted_size", metrics.unweighted_size);
    set(ret, "tthresh:stop_ranks_size", metrics.stop_ranks_size);
    set(ret, "tthresh:slice_norm_size", metrics.slice_norm_size);
    return ret;
  }

  double target_value = 0;
  Target target = Target::eps;
  tthresh_metric metrics;
};


static pressio_register tthresh_compressor_plugin_register(compressor_plugins(), "tthresh", [] {
      return compat::make_unique<tthresh_compressor_plugin>();
    }
);
