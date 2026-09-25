from typing import Hashable, Literal

import numpy as np
import xarray as xr


@xr.register_dataarray_accessor("bitpacker")
class BitPacker:
    """DataArray extension allowing bitpacking of boolean arrays"""

    def __init__(self, xarray_obj):
        """"""
        self._obj = xarray_obj

    def packbits(
        self,
        # array: xr.DataArray,
        dim: Hashable,
        bitorder: Literal["big", "little"] = "big",
        separator: str = " | ",
    ) -> xr.DataArray:
        """
        Pack boolean array along given dimension using numpy packbits

        The DataArray this is called on must be a boolean array containing multiple
        flags to be packed. The dimension listed in "dim" must index across the flags
        to be packed. If "dim" has a corresponding coordinate of the same type, it will
        be coerced to a string dtype and used as the name for each flag. No name (after
        coercion) should contain the value of "separator" as a substring.

        args:
            dim (hashable): Dimension of array over which bit packing should be
                done.
            bitorder ("big" or "little"): The order in which to pack bits; see the docs
                for numpy.packbits for more info.
            separator (string): String used to separate flag names in the bitpacked
                metadata. Default is " | ".

        returns:
            xarray.DataArray: Unsigned 8-bit integer array with the same shape and
                dimensions as the input array, except for the dimension "dim" which is
                removed. Each integer has the flags for that location bit-packed using
                numpy.bitpack. The "long_name" attribute will have "Bit-packed"
                prepended if present, and will be created as "Bit-packed flags" if not.
                The bitorder used is saved as the attribute "bitorder", and the
                attribute "bit_flags" contains the flag names (extracted from the
                coordinates on dim in array) separated by the separator value.
        """
        try:
            axis = list(self._obj.dims).index(dim)
        except ValueError as e:
            e.add_note(f"dim must be a dimension of array.")
            raise
        if self._obj.coords[dim].dtype.type is not np.str_:
            raise ValueError(
                f"Dimension '{dim}' must have string dtype; got {self._obj.coords[dim].dtype}"
            )
        packed_data = np.packbits(self._obj.values, axis=axis, bitorder=bitorder)
        output_array = xr.DataArray(
            packed_data, dims=self._obj.dims, attrs=self._obj.attrs
        )
        output_array = output_array.squeeze(dim=dim, drop=True)
        try:
            output_array.attrs["long_name"] = (
                "Bit-packed " + self._obj.attrs["long_name"]
            )
        except KeyError:
            output_array.attrs["long_name"] = "Bit-packed flags"
        output_array.attrs["bitorder"] = bitorder
        output_array.attrs["bit_flags"] = separator.join(self._obj.coords[dim].values)
        output_array.attrs["bit_flag_separator"] = separator
        output_array.attrs["valid_range"] = [0, 255]
        return output_array

    def unpackbits(
        self,
        # array: xr.DataArray,
        dim: Hashable,
        axis: int = -1,
    ) -> xr.DataArray:
        """
        Unpack bit-packed DataArray into a boolean array

        The DataArray this is called on should be a bit-packed array of unsigned 8-bit
        integers in the format output by the "packbits" method of this class. Should
        have a "bitorder" attribute containing the bit ordering for the packing, either
        "big" or "little", a "bit_flags" attribute which contains the flag names
        separated by the substring found in the "big_flag_separator" attribute, and a
        "long_name" attribute starting with "Bit-packed ".

        args:
            dim (hashable): Dimension to be added to index between the unpacked flags.
            axis (int): Position in the list of dimensions where the new dimension
                should be added. Default is -1 (end of the list).

        returns:
            xarray.DataArray: Integer binary array of the same shape as the input array,
                but with a new dimension "dim" added at position "axis". For each
                position along this dimension, the binary value represents the unpacked
                status of the corresponding bit. The dimension "dim" has a coordinate
                list of flag names extracted from the attribute "bit_flags" in the input
                array.
        """
        # Insert the new dimension at the appropriate location
        unpacked_dims = list(self._obj.dims)
        if axis >= 0:
            unpacked_dims.insert(axis, dim)
        else:  # Insert doesn't handle negative indices the same way as numpy
            unpacked_dims.insert(len(self._obj.dims) + 1 + axis, dim)

        # Unpack the coordinates using our known separator
        unpacked_coords = self._obj.attrs["bit_flags"].split(
            self._obj.attrs["bit_flag_separator"]
        )

        # Unpack the data, using the length of our coordinates to not excessively 0-pad
        unsqueezed_data = np.expand_dims(self._obj.values, axis)
        unpacked_data = np.unpackbits(
            unsqueezed_data,
            axis=axis,
            bitorder=self._obj.attrs["bitorder"],
            count=len(unpacked_coords),
        )

        # Package everything into an array
        unpacked_array = xr.DataArray(
            unpacked_data,
            dims=unpacked_dims,
            coords={dim: unpacked_coords},
            attrs=self._obj.attrs,
        )

        # Adjust array metadata
        # Remove "Bit-packed " from the beginning of the long name
        unpacked_array.attrs["long_name"] = unpacked_array.attrs["long_name"][11:]
        # Reset valid range
        unpacked_array.attrs["valid_range"] = [0, 1]
        # Remove no longer relevant flags
        del unpacked_array.attrs["bitorder"]
        del unpacked_array.attrs["bit_flags"]
        del unpacked_array.attrs["bit_flag_separator"]
        return unpacked_array
