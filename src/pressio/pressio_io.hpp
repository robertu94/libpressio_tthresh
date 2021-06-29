/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __IO_HPP__
#define __IO_HPP__

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <sstream>
//#include "zlib.h"

// Avoids corruption of the input and output data on Windows/MS-DOS systems
#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#  include <fcntl.h>
#  include <io.h>
#  define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#  define SET_BINARY_MODE(file)
#endif

struct {
    uint64_t rbytes, wbytes;
    int8_t rbit, wbit;
    std::stringstream file; // File handle to read/write from/to
    uint8_t inout[CHUNK]; // Buffer to write the results of inflation/deflation
    uint8_t buf[CHUNK]; // Buffer used for the read/write operations
    int32_t bufstart = 0;
    int32_t bufend = 0;
    size_t total_written_bytes = 0; // Used to compute the final file size
} zs; // Read/write state for zlib interfacing

void debug_io() {
  std::cout << "last gstatus =" << zs.file.gcount() << std::endl;
  std::cout << "total gwritten =" << zs.file.tellg() << std::endl;
  std::cout << "total pwritten =" << zs.file.tellp() << std::endl;
  std::cout << "is good? " << std::boolalpha << zs.file.good() << std::endl;
}

std::string dump_io() {
  return zs.file.str();
}


/*********/
// Writing
/*********/

// Call open_wbit() before write_bits()
// If write_bits() has been called, call close_wbit() before write_stream()

void open_write() {
    zs.file = std::stringstream(std::ios_base::in | std::ios_base::out | std::ios_base::binary);
}

void write_stream(unsigned char *buf, size_t bytes_to_write)
{
    zs.file.write(reinterpret_cast<const char*>(buf), bytes_to_write);
    zs.total_written_bytes += bytes_to_write;
}

void open_wbit() {
    zs.wbytes = 0;
    zs.wbit = 63;
}

// Assumption: to_write <= 64
void write_bits(uint64_t bits, char to_write) {
    if (to_write <= zs.wbit+1) {
        zs.wbytes |= bits << (zs.wbit+1-to_write);
        zs.wbit -= to_write;
    }
    else {
        if (zs.wbit > -1)
            zs.wbytes |= bits >> (to_write-(zs.wbit+1));
        write_stream(reinterpret_cast<unsigned char *> (&zs.wbytes), sizeof(zs.wbytes));
        to_write -= zs.wbit+1;
        zs.wbytes = 0;
        zs.wbytes |= bits << (64-to_write);
        zs.wbit = 63-to_write;
    }
}

void close_wbit() {
    // Write any reamining bits
    if (zs.wbit < 63)
        write_stream(reinterpret_cast < unsigned char *> (&zs.wbytes), sizeof(zs.wbytes));
}

void close_write() {
}

/*********/
// Reading
/*********/

// If read_bits() has been called, call close_rbit() before read_stream()

void open_read(std::string&& contents)
{
    zs.file = std::stringstream(contents, std::ios::in | std::ios::out | std::ios::binary);
    zs.rbytes = 0;
    zs.rbit = -1;
}

void read_stream(uint8_t *buf, size_t bytes_to_read)
{
    zs.file.read(reinterpret_cast<char*>(buf), bytes_to_read);
    size_t howmany = zs.file.gcount();
    if (howmany != bytes_to_read) {
        std::cout << "Error: tried to read " << bytes_to_read << " bytes, got only " << howmany << std::endl;
        exit(1);
    }
}

void close_rbit()
{
    zs.rbytes = 0;
    zs.rbit = -1;
}

// Assumption: to_read <= BITS
uint64_t read_bits(char to_read) {
    uint64_t result = 0;
    if (to_read <= zs.rbit+1) {
        result = zs.rbytes << (63-zs.rbit) >> (64-to_read);
        zs.rbit -= to_read;
    }
    else {
        if (zs.rbit > -1)
            result = zs.rbytes << (64-zs.rbit-1) >> (64-to_read);
        read_stream(reinterpret_cast<uint8_t *> (&zs.rbytes), sizeof(zs.rbytes));
        to_read -= zs.rbit+1;
        result |= zs.rbytes >> (64-to_read);
        zs.rbit = 63-to_read;
    }
    return result;
}

void close_read()
{
}

#endif // IO_HPP
