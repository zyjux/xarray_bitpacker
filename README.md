# Xarray Bitpacker
An xarray DataArray accessor extension that leverages xarray's metadata to automate and enhance numpy's bitpacking functionality. 

When saving data that consists of multiple boolean or binary arrays, it can be more efficient to "bit-pack" the data by recording the state of each flag as the state of a particular bit in a standard 8-bit unsigned integer.
This allows up to 8 boolean fields to be encoded into a single integer field with no loss of information.
However, while this allows for efficient storage, the bitpacked fields are not as intuitive to use and plot as the separate binary fields, so both bitpacking and unbitpacking routines are needed.
Numpy provides efficient bitpacking and unbitpacking routines, but these require that the information about what flag each bit represents and how the bitpacking was performed (including whether the bits are ordered from big to little or little to big) must be recorded separately.
This package makes use of the enhanced metadata structures available in xarray to store all that information alongside the bitpacked array, allowing for easy and automated unbitpacking.

We follow the "accessor" extension framework for xarray found [here](https://docs.xarray.dev/en/stable/internals/extending-xarray.html) to patch in these routines as DataArray methods.

## Usage
When imported, adds the `bitpacker` namespace to `xarray.DataArrays`, which contains `packbits` and `unpackbits`.
`packbits` takes a set of boolean or binary flag arrays structured as a single, higher-dimensional DataArray with one dimension that indexes across the flags and bitpacks that information as an 8-bit integer array, with metadata describing the packing process and flags.
`unpackbits` then takes a packed array with metadata and unpacks it back into a higher-dimensional array with coordinates specifying the packed flags.

```
import xarray as xr
import numpy as np

import xarray_bitpacker

da = xr.DataArray(
  np.stack([np.full((5,), False), np.full((5,), True)], axis=0),
  dims=["flags", "x"],
  coords={"flags": ["flag_1", "flag_2"]}
)

packed_array = da.bitpacker.packbits(dim="flags", bitorder="little")
packed_array

>> <xarray.DataArray (x: 5)> Size: 5B
>> array([2, 2, 2, 2, 2], dtype=uint8)
>> Dimensions without coordinates: x
>> Attributes:
>>     long_name:           Bit-packed flags
>>     bitorder:            little
>>     bit_flags:           flag_1 | flag_2
>>     bit_flag_separator:   | 
>>     valid_range:         [0, 255]

unpacked_array = packed_array.bitpacker.unpackbits(dim="flags")
unpacked_array

>> <xarray.DataArray (x: 5, flags: 2)> Size: 10B
>> array([[0, 1],
>>        [0, 1],
>>        [0, 1],
>>        [0, 1],
>>        [0, 1]], dtype=uint8)
>> Coordinates:
>>   * flags    (flags) <U6 48B 'flag_1' 'flag_2'
>> Dimensions without coordinates: x
>> Attributes:
>>     long_name:    flags
>>     valid_range:  [0, 1]
```

## AI Statement

No generative AI of any kind was used for development, testing, or coding assistance on this project.

## Acknowledgments

The development of this package was supported by the Cooperative Institute for Research in the Atmosphere (CIRA) at Colorado State University and was funded by the U.S. Office of Naval Research (ONR) via the OVERCAST contract, Award N0001424C2214.
