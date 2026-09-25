# xarray_bitpacker
An xarray DataArray accessor extension that leverages xarray's metadata to automate and enhance numpy's bitpacking functionality. 

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
