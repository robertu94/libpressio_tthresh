#ifndef TTHRESH_TTHRESH_LIB_H
#define TTHRESH_TTHRESH_LIB_H

// Size (in bytes) for all I/O buffers
#define CHUNK (1<<18)

// Rows whose squared norm is larger than this will be cropped away
#define AUTOCROP_THRESHOLD (1e-10)

#include <vector>
#include "tthresh_metrics.h"
#include "pressio_encode.hpp"
#include "tthresh.hpp"
#include "tucker.hpp"
#include "pressio_io.hpp"
#include "decode.hpp"
#include <unistd.h>
#include <math.h>
#include <Eigen/Dense>
#include <map>
#include <libpressio_ext/cpp/data.h>


typedef __float128 LLDOUBLE;
typedef __float80 LDOUBLE;

using namespace std;
using namespace Eigen;

int qneeded;

double rle_time = 0;
double raw_time = 0;

double price = -1, total_bits_core = -1, eps_core = -1;
size_t total_bits = 0;


vector<uint64_t> encode_array(double* c, size_t size, double eps_target, bool is_core) {

  /**********************************************/
  // Compute and save maximum (in absolute value)
  /**********************************************/

  double maximum = 0;
  for (size_t i = 0; i < size; i++) {
    if (abs(c[i]) > maximum)
      maximum = abs(c[i]);
  }
  double scale = ldexp(1, 63-ilogb(maximum));

  uint64_t tmp;
  memcpy(&tmp, (void*)&scale, sizeof(scale));
  write_bits(tmp, 64);

  LLDOUBLE normsq = 0;
  vector<uint64_t> coreq(size);

  // 128-bit float arithmetics are slow, so we split the computation of normsq into partial sums
  size_t stepsize = 100;
  size_t nsteps = ceil(size/double(stepsize));
  size_t pos = 0;
  for (size_t i = 0; i < nsteps; ++i) {
    LDOUBLE partial_normsq = 0;
    for (size_t j = 0; j < stepsize; ++j) {
      coreq[pos] = uint64_t(abs(c[pos])*scale);
      partial_normsq += LDOUBLE(abs(c[pos]))*abs(c[pos]);
      pos++;
      if (pos == size)
        break;
    }
    normsq += partial_normsq;
    if (pos == size)
      break;
  }
  normsq *= LLDOUBLE(scale)*LLDOUBLE(scale);

  LLDOUBLE sse = normsq;
  LDOUBLE last_eps = 1;
  LDOUBLE thresh = eps_target*eps_target*normsq;

  /**************/
  // Encode array
  /**************/

  vector<uint64_t> current(size, 0);

  bool done = false;
  total_bits = 0;
  size_t last_total_bits = total_bits;
  double eps_delta = 0, size_delta = 0, epsilon;
  int q;
  bool all_raw = false;
  for (q = 63; q >= 0; --q) {
    vector<uint64_t> rle;
    LDOUBLE plane_sse = 0;
    size_t plane_ones = 0;
    size_t counter = 0;
    size_t i;
    vector<bool> raw;
    for (i = 0; i < size; ++i) {
      bool current_bit = ((coreq[i]>>q)&1UL);
      plane_ones += current_bit;
      if (not all_raw and current[i] == 0) { // Feed to RLE
        if (not current_bit)
          counter++;
        else {
          rle.push_back(counter);
          counter = 0;
        }
      }
      else { // Feed to raw stream
        ++total_bits;
        raw.push_back(current_bit);
      }

      if (current_bit) {
        plane_sse += (LDOUBLE(coreq[i] - current[i]));
        current[i] |= 1UL<<q;
        if (plane_ones%100 == 0) {
          LDOUBLE k = 1UL<<q;
          LDOUBLE sse_now = sse+(-2*k*plane_sse + k*k*plane_ones);
          if (sse_now <= thresh) {
            done = true;
            break;
          }
        }

      }
    }

    LDOUBLE k = 1UL<<q;
    sse += -2*k*plane_sse + k*k*plane_ones;
    rle.push_back(counter);

    uint64_t rawsize = raw.size();
    write_bits(rawsize, 64);
    total_bits += 64;

    {
      high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
      for (size_t i = 0; i < raw.size(); ++i)
        write_bits(raw[i], 1);
      raw_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
    }
    {
      high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
      uint64_t this_part = encode(rle);
      rle_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
      total_bits += this_part;
    }

    epsilon = sqrt(double(sse/normsq));
    if (last_total_bits > 0) {
      if (is_core) {
        size_delta = (total_bits - last_total_bits) / double(last_total_bits);
        eps_delta = (last_eps - epsilon) / epsilon;
      }
      else {
        if ((total_bits/total_bits_core) / (epsilon/eps_core) >= price)
          done = true;
      }
    }
    last_total_bits = total_bits;
    last_eps = epsilon;

    if (raw.size()/double(size) > 0.8)
      all_raw = true;

    write_bits(all_raw, 1);
    total_bits++;

    write_bits(done, 1);
    total_bits++;

    if (done)
      break;
  }

  //TODO METRICS stop_timer

  /****************************************/
  // Save signs of significant coefficients
  /****************************************/

  for (size_t i = 0; i < size; ++i) {
    if (current[i] > 0) {
      write_bits((c[i] > 0), 1);
      total_bits++;
    }
  }

  if (is_core) {
    price = size_delta / eps_delta;
    eps_core = epsilon;
    total_bits_core = total_bits;
  }
  return current;
}

double *compress_pressio_tthresh(pressio_data const* input, pressio_data* compressed, Target target, double target_value, tthresh_metric& metrics) {

  n = s.size();
  cumulative_products(s, sprod);

  /***********************/
  // Check input data type
  /***********************/

  size_t size = sprod[n]; // Total number of tensor elements
  uint8_t io_type_size, io_type_code;
  const pressio_dtype input_dtype = pressio_data_dtype(input);
  switch(input_dtype) {
    case pressio_uint8_dtype:
    case pressio_byte_dtype:
      io_type_size = pressio_dtype_size(input_dtype);
      io_type_code = 0;
      break;
  case pressio_uint16_dtype:
      io_type_size = pressio_dtype_size(input_dtype);
      io_type_code = 1;
      break;
  case pressio_int32_dtype:
      io_type_size = pressio_dtype_size(input_dtype);
      io_type_code = 2;
      break;
  case pressio_float_dtype:
      io_type_size =  pressio_dtype_size(input_dtype);
      io_type_code = 3;
      break;
  case pressio_double_dtype:
      io_type_size =  pressio_dtype_size(input_dtype);
      io_type_code = 4;
      break;
  default:
    throw std::runtime_error("unsupported type");
  }

  /************************/
  // Check input file sizes
  /************************/

  size_t expected_size = size * io_type_size;
  size_t fsize = pressio_data_get_bytes(input);
  if (expected_size != fsize) {
    throw std::runtime_error("expected size did not equal buffer size");
  }

  /********************************************/
  // Save tensor dimensionality, sizes and type
  /********************************************/

  open_write();
  auto inital_pos = zs.file.tellp();
  write_stream(reinterpret_cast < unsigned char *> (&n), sizeof(n));
  write_stream(reinterpret_cast < unsigned char *> (&s[0]), n*sizeof(s[0]));
  write_stream(reinterpret_cast < unsigned char *> (&io_type_code), sizeof(io_type_code));

  /*****************************/
  // Load input file into memory
  /*****************************/

  //TODO TIMER start loading and casting data
  char *in = static_cast<char*>(pressio_data_ptr(input, NULL));

  // Cast the data to doubles
  pressio_data* double_input = pressio_data_cast(input, pressio_double_dtype);
  double *data = static_cast<double*>(pressio_data_ptr(double_input, NULL));
  double datamin = numeric_limits < double >::max(); // Tensor statistics
  double datamax = numeric_limits < double >::lowest();
  double datanorm = 0;

  for (size_t i = 0; i < size; ++i) {
    datamin = min(datamin, data[i]); // Compute statistics, since we're at it
    datamax = max(datamax, data[i]);
    datanorm += data[i] * data[i];
  }
  datanorm = sqrt(datanorm);

  /**********************************************************************/
  // Compute the target SSE (sum of squared errors) from the given metric
  /**********************************************************************/

  double sse;
  if (target == eps)
    sse = pow(target_value * datanorm, 2);
  else if (target == rmse)
    sse = pow(target_value, 2) * size;
  else
    sse = pow((datamax - datamin) / (2 * (pow(10, target_value / 20))), 2) * size;
  double epsilon = sqrt(sse) / datanorm;

  /*********************************/
  // Create and decompose the tensor
  /*********************************/

  //TODO metrics tucker decomposition
  double *c = new double[size]; // Tucker core

  memcpy(c, data, size * sizeof(double));

  vector<MatrixXd> Us(n); // Tucker factor matrices
  hosvd_compress(c, Us, false);

  //TODO stop metrics tucker decomposition

  /**************************/
  // Encode and save the core
  /**************************/

  open_wbit();
  vector<uint64_t> current = encode_array(c, size, epsilon, true);
  close_wbit();
  const auto encode_end_pos = zs.file.tellp();
  metrics.encoded_core_size = encode_end_pos - inital_pos;

  /*******************************/
  // Compute and save tensor ranks
  /*******************************/

  //TODO compute ranks
  r = vector<uint32_t> (n, 0);
  vector<size_t> indices(n, 0);
  vector< RowVectorXd > slicenorms(n);
  for (int dim = 0; dim < n; ++dim) {
    slicenorms[dim] = RowVectorXd(s[dim]);
    slicenorms[dim].setZero();
  }
  for (size_t i = 0; i < size; ++i) {
    if (current[i] > 0) {
      for (int dim = 0; dim < n; ++dim) {
        slicenorms[dim][indices[dim]] += double(current[i])*current[i];
      }
    }
    indices[0]++;
    int pos = 0;
    while (indices[pos] >= s[pos] and pos < n-1) {
      indices[pos] = 0;
      pos++;
      indices[pos]++;
    }
  }

  for (int dim = 0; dim < n; ++dim) {
    for (size_t i = 0; i < s[dim]; ++i) {
      if (slicenorms[dim][i] > 0)
        r[dim] = i+1;
      slicenorms[dim][i] = sqrt(slicenorms[dim][i]);
    }
  }
  //TODO metrics stop ranks

  write_stream(reinterpret_cast<unsigned char*> (&r[0]), n*sizeof(r[0]));
  metrics.stop_ranks_size = n * sizeof(r[0]);

  for (uint8_t i = 0; i < n; ++i) {
    write_stream(reinterpret_cast<uint8_t*> (slicenorms[i].data()), r[i]*sizeof(double));
    metrics.slice_norm_size += r[i]*sizeof(double);
  }

  vector<MatrixXd> Uweighteds;
  const auto unweighted_begin = zs.file.tellp();
  open_wbit();
  for (int dim = 0; dim < n; ++dim) {
    MatrixXd Uweighted = Us[dim].leftCols(r[dim]);
    for (size_t col = 0; col < r[dim]; ++col)
      Uweighted.col(col) = Uweighted.col(col)*slicenorms[dim][col];
    Uweighteds.push_back(Uweighted);
    encode_array(Uweighted.data(), s[dim]*r[dim], 0, false);//*(s[i]*s[i]/sprod[n]));  // TODO flatten in F order?
  }
  close_wbit();
  const auto final_pos = zs.file.tellp();
  metrics.unweighted_size = final_pos - unweighted_begin;
  const size_t compressed_size =  final_pos - inital_pos;
  *compressed = pressio_data::owning(pressio_byte_dtype, {compressed_size});
  zs.file.read(
      static_cast<char*>(compressed->data()),
      compressed->size_in_bytes()
      );
  close_write();
  delete[] c;
  size_t newbits = zs.total_written_bytes * 8;
  return data;
}


double maximum;
int q;
size_t pointer;

/////
double decode_rle_time;
double decode_raw_time;
double unscramble_time;
/////

vector<uint64_t> decode_array(size_t size, bool is_core) {

  uint64_t tmp = read_bits(64);
  memcpy(&maximum, (void*)&tmp, sizeof(tmp));

  vector<uint64_t> current(size, 0);

  decode_rle_time = 0;
  decode_raw_time = 0;
  unscramble_time = 0;

  int zeros = 0;
  bool all_raw = false;
  for (q = 63; q >= 0; --q) {
    uint64_t rawsize = read_bits(64);

    size_t read_from_rle = 0;
    size_t read_from_raw = 0;

    if (all_raw) {
      high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
      for (uint64_t pointer = 0; pointer < rawsize; ++pointer) {
        current[pointer] |= read_bits(1) << q;
      }
      unscramble_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
      vector<size_t> rle;
      decode(rle);
    }
    else {
      vector<bool> raw;
      high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
      for (uint64_t i = 0; i < rawsize; ++i)
        raw.push_back(read_bits(1));
      decode_raw_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;

      vector<size_t> rle;
      timenow = chrono::high_resolution_clock::now();
      decode(rle);
      decode_rle_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;

      int64_t raw_index = 0;
      int64_t rle_value = -1;
      int64_t rle_index = -1;

      timenow = chrono::high_resolution_clock::now();
      for (pointer = 0; pointer < size; ++pointer) {
        uint64_t this_bit = 0;
        if (not all_raw and current[pointer] == 0) { // Consume bit from RLE
          if (rle_value == -1) {
            rle_index++;
            if (rle_index == int64_t(rle.size()))
              break;
            rle_value = rle[rle_index];
          }
          if (rle_value >= 1) {
            read_from_rle++;
            this_bit = 0;
            rle_value--;
          }
          else if (rle_value == 0) {
            read_from_rle++;
            this_bit = 1;
            rle_index++;
            if (rle_index == int64_t(rle.size()))
              break;
            rle_value = rle[rle_index];
          }
        }
        else { // Consume bit from raw
          if (raw_index == int64_t(raw.size()))
            break;
          this_bit = raw[raw_index];
          read_from_raw++;
          raw_index++;
        }
        if (this_bit)
          current[pointer] |= this_bit << q;
      }
      unscramble_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
    }

    all_raw = read_bits(1);

    bool done = read_bits(1);
    if (done)
      break;
    else
      zeros++;
  }
  return current;
}

vector<double> dequantize(vector<uint64_t>& current) { // TODO after resize
  size_t size = current.size();
  vector<double> c(size, 0);
  for (size_t i = 0; i < size; ++i) {
    if (current[i] > 0) {
      if (i < pointer) {
        if (q >= 1)
          current[i] += 1UL<<(q-1);
      }
      else
        current[i] += 1UL<<q;
      char sign = read_bits(1);
      c[i] = double(current[i]) / maximum * (sign*2-1);
    }
  }
  return c;
}

void decompress_pressio_tthresh(pressio_data const* compressed, pressio_data* output, vector<Slice>& cutout, bool autocrop, tthresh_metric& metrics) {

  /***************************************************/
  // Read output tensor dimensionality, sizes and type
  /***************************************************/

  std::string compressed_file(static_cast<char*>(compressed->data()), compressed->size_in_bytes());
  open_read(std::move(compressed_file));
  read_stream(reinterpret_cast<uint8_t*> (&n), sizeof(n));
  s = vector<uint32_t> (n);
  read_stream(reinterpret_cast<uint8_t*> (&s[0]), n * sizeof(s[0]));

  bool whole_reconstruction = cutout.size() == 0;
  if (cutout.size() < n) // Non-specified slicings are assumed to be the standard (0,1,-1)
    for (uint32_t j = cutout.size(); j < s.size(); ++j)
      cutout.push_back(Slice(0, -1, 1));

  cumulative_products(s, sprod);
  size_t size = sprod[n];
  snew = vector<uint32_t> (n);
  for (uint8_t i = 0; i < n; ++i) {
    cutout[i].update(s[i]);
    snew[i] = cutout[i].get_size();
  }
  cumulative_products(snew, snewprod);


  uint8_t io_type_code;
  read_stream(reinterpret_cast<uint8_t*> (&io_type_code), sizeof(io_type_code));
  uint8_t io_type_size;
  pressio_dtype output_type;
  if (io_type_code == 0) {
    io_type_size = sizeof(unsigned char);
    output_type = pressio_uint8_dtype;
  }
  else if (io_type_code == 1) {
    io_type_size = sizeof(unsigned short);
    output_type = pressio_uint16_dtype;
  }
  else if (io_type_code == 2) {
    io_type_size = sizeof(int);
    output_type = pressio_int32_dtype;
  }
  else if (io_type_code == 3) {
    io_type_size = sizeof(float);
    output_type = pressio_float_dtype;
  }
  else {
    io_type_size = sizeof(double);
    output_type = pressio_double_dtype;
  }

  /*************/
  // Decode core
  /*************/

  vector<uint64_t> current = decode_array(sprod[n], true);
  vector<double> c = dequantize(current);
  close_rbit();

  /*******************/
  // Read tensor ranks
  /*******************/

  r = vector<uint32_t> (n);
  read_stream(reinterpret_cast<uint8_t*> (&r[0]), n*sizeof(r[0]));
  rprod = vector<size_t> (n+1);
  rprod[0] = 1;
  for (uint8_t i = 0; i < n; ++i)
    rprod[i+1] = rprod[i]*r[i];

  vector<RowVectorXd> slicenorms(n);
  for (uint8_t i = 0; i < n; ++i) {
    slicenorms[i] = RowVectorXd(r[i]);
    for (uint64_t col = 0; col < r[i]; ++col) { // TODO faster
      double norm;
      read_stream(reinterpret_cast<uint8_t*> (&norm), sizeof(double));
      slicenorms[i][col] = norm;
    }
  }

  //**********************/
  // Reshape core in place
  //**********************/

  size_t index = 0; // Where to read from in the original core
  vector<size_t> indices(n, 0);
  uint8_t pos = 0;
  for (size_t i = 0; i < rprod[n]; ++i) { // i marks where to write in the new rank-reduced core
    c[i] = c[index];
    indices[0]++;
    index++;
    pos = 0;
    // We update all necessary indices in cascade, left to right. pos == n-1 => i == rprod[n]-1 => we are done
    while (indices[pos] >= r[pos] and pos < n-1) {
      indices[pos] = 0;
      index += sprod[pos+1] - r[pos]*sprod[pos];
      pos++;
      indices[pos]++;
    }
  }

  //*****************/
  // Reweight factors
  //*****************/

  vector< MatrixXd > Us;
  for (uint8_t i = 0; i < n; ++i) {
    vector<uint64_t> factorq = decode_array(s[i]*r[i], false);
    vector<double> factor = dequantize(factorq);
    MatrixXd Uweighted(s[i], r[i]);
    memcpy(Uweighted.data(), (void*)factor.data(), sizeof(double)*s[i]*r[i]);
    MatrixXd U(s[i], r[i]);
    for (size_t col = 0; col < r[i]; ++col) {
      if (slicenorms[i][col] > 1e-10)
        U.col(col) = Uweighted.col(col)/slicenorms[i][col];
      else
        U.col(col) *= 0;
    }
    Us.push_back(U);
  }
  close_rbit();

  /*************************/
  // Autocrop (if requested)
  /*************************/

  if (autocrop) {
    cout << "autocrop =";
    for (uint8_t dim = 0; dim < n; ++dim) {
      uint32_t start_row = 0, end_row = 0;
      bool start_set = false;
      for (int i = 0; i < Us[dim].rows(); ++i) {
        double sqnorm = 0;
        for (int j = 0; j < Us[dim].cols(); ++j)
          sqnorm += Us[dim](i,j)*Us[dim](i,j);
        if (sqnorm > AUTOCROP_THRESHOLD) {
          if (not start_set) {
            start_row = i;
            start_set = true;
          }
          end_row = i+1;
        }
      }
      cutout[dim].points[0] = start_row;
      cutout[dim].points[1] = end_row;
      snew[dim] = end_row-start_row;
      cout << " " << start_row << ":" << end_row;
    }
    cout << endl;
    cumulative_products(snew, snewprod);
  }

  /************************/
  // Reconstruct the tensor
  /************************/

  hosvd_decompress(c, Us, false, cutout);

  /***********************************/
  // Cast and write the result on disk
  /***********************************/

  std::stringstream output_stream(ios::out | std::ios::in | ios::binary);
  size_t buf_elems = CHUNK;
  vector<uint8_t> buffer(io_type_size * buf_elems);
  size_t buffer_wpos = 0;
  double sse = 0;
  double datanorm = 0;
  double datamin = std::numeric_limits < double >::max();
  double datamax = std::numeric_limits < double >::min();
  double remapped = 0;
  for (size_t i = 0; i < snewprod[n]; ++i) {
    if (io_type_code == 0) {
      remapped = (unsigned char)(std::round(std::max<uint64_t>(0.0, min<uint64_t>(double(std::numeric_limits<unsigned char>::max()), c[i]))));
      reinterpret_cast < unsigned char *>(&buffer[0])[buffer_wpos] = remapped;
    }
    else if (io_type_code == 1) {
      remapped = (unsigned short)(std::round(std::max<uint64_t>(0.0, min<uint64_t>(double(std::numeric_limits<unsigned short>::max()), c[i]))));
      reinterpret_cast < unsigned short *>(&buffer[0])[buffer_wpos] = remapped;
    }
    else if (io_type_code == 2) {
      remapped = int(std::round(std::max<uint64_t>(std::numeric_limits<int>::min(), min<uint64_t>(double(std::numeric_limits<int>::max()), c[i]))));;
      reinterpret_cast < int *>(&buffer[0])[buffer_wpos] = remapped;
    }
    else if (io_type_code == 3) {
      remapped = float(c[i]);
      reinterpret_cast < float *>(&buffer[0])[buffer_wpos] = remapped;
    }
    else {
      remapped = c[i];
      reinterpret_cast < double *>(&buffer[0])[buffer_wpos] = remapped;
    }
    buffer_wpos++;
    if (buffer_wpos == buf_elems) {
      buffer_wpos = 0;
      output_stream.write(reinterpret_cast<const char*>(&buffer[0]), io_type_size * buf_elems);
    }
  }
  if (buffer_wpos > 0)
    output_stream.write(reinterpret_cast<const char*>(&buffer[0]), io_type_size * buffer_wpos);
  *output = pressio_data::owning(
      output_type,
      output->dimensions()
      );

  output_stream.read(static_cast<char*>(output->data()), output->size_in_bytes());
}
#endif //TTHRESH_TTHRESH_LIB_H
