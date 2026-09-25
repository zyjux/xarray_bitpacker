import unittest

import numpy as np
import xarray as xr

import xarray_bitpacker  # type: ignore


class Test_packbits(unittest.TestCase):
    def test_AllFalseArray_AllZeroPackedArray(self):
        fake_bool_data = np.full((2, 5), False)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["flag", "y"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        mock_result = xr.DataArray(np.zeros((5,), dtype=int), dims=["y"])
        xr.testing.assert_equal(result, mock_result)

    def test_AllZerosArray_AllZeroPackedArray(self):
        fake_bool_data = np.full((2, 5), 0)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["flag", "y"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        mock_result = xr.DataArray(np.zeros((5,), dtype=int), dims=["y"])
        xr.testing.assert_equal(result, mock_result)

    def test_AllTrueArray_PackedArrayAll192(self):
        fake_bool_data = np.full((2, 5), True)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["flag", "y"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        mock_result = xr.DataArray(192 * np.ones((5,), dtype=int), dims=["y"])
        xr.testing.assert_equal(result, mock_result)

    def test_AllOnesArray_PackedArrayAll192(self):
        fake_bool_data = np.full((2, 5), 1)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["flag", "y"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        mock_result = xr.DataArray(192 * np.ones((5,), dtype=int), dims=["y"])
        xr.testing.assert_equal(result, mock_result)

    def test_FirstFlagAllTrueSecondAllFalse_All128(self):
        fake_bool_data = np.stack([np.full((5,), True), np.full((5,), False)], axis=0)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["flag", "y"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        mock_result = xr.DataArray(128 * np.ones((5,), dtype=int), dims=["y"])
        xr.testing.assert_equal(result, mock_result)

    def test_FirstFlagAllFalseSecondAllTrue_All64(self):
        fake_bool_data = np.stack([np.full((5,), False), np.full((5,), True)], axis=0)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["flag", "y"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        mock_result = xr.DataArray(64 * np.ones((5,), dtype=int), dims=["y"])
        xr.testing.assert_equal(result, mock_result)

    def test_FirstFlagAllFalseSecondAllTrueBitorderLittle_All2(self):
        fake_bool_data = np.stack([np.full((5,), False), np.full((5,), True)], axis=0)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["flag", "y"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag", bitorder="little")
        mock_result = xr.DataArray(2 * np.ones((5,), dtype=int), dims=["y"])
        xr.testing.assert_equal(result, mock_result)

    def test_DimNotInListOfDims_RaiseException(self):
        fake_bool_data = np.full((2, 5), False)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["flag", "y"], coords={"flag": ["flag1", "flag2"]}
        )
        with self.assertRaises(ValueError):
            _ = fake_array.bitpacker.packbits("flags")

    def test_DimNot0thInListOfDimsAllFalse_AllZero(self):
        fake_bool_data = np.full((5, 2), False)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["y", "flag"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        mock_result = xr.DataArray(np.zeros((5,), dtype=int), dims=["y"])
        xr.testing.assert_equal(result, mock_result)

    def test_DimNot0thInListOfDimsAllTrue_All192(self):
        fake_bool_data = np.full((5, 2), True)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["y", "flag"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        mock_result = xr.DataArray(np.full((5,), 192, dtype=int), dims=["y"])
        xr.testing.assert_equal(result, mock_result)

    def test_CommaSeparator_BitFlagsAndSeparatorAttrsCorrect(self):
        fake_bool_data = np.full((5, 2), False)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["y", "flag"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag", separator=", ")
        with self.subTest("Bit Flags"):
            self.assertEqual(result.attrs["bit_flags"], "flag1, flag2")
        with self.subTest("Separator"):
            self.assertEqual(result.attrs["bit_flag_separator"], ", ")

    def test_NoInputArrayAttrs_CorrectAttrs(self):
        fake_bool_data = np.full((5, 2), False)
        fake_array = xr.DataArray(
            fake_bool_data, dims=["y", "flag"], coords={"flag": ["flag1", "flag2"]}
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        mock_attrs = {
            "long_name": "Bit-packed flags",
            "bitorder": "big",
            "bit_flags": "flag1 | flag2",
            "bit_flag_separator": " | ",
            "valid_range": [0, 255],
        }
        self.assertDictEqual(result.attrs, mock_attrs)

    def test_InputArrayHasLongNameAttr_PrependBitPackedToLongname(self):
        fake_bool_data = np.full((5, 2), False)
        fake_array = xr.DataArray(
            fake_bool_data,
            dims=["y", "flag"],
            coords={"flag": ["flag1", "flag2"]},
            attrs={"long_name": "Test Flags"},
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        self.assertEqual(result.attrs["long_name"], "Bit-packed Test Flags")

    def test_InputArrayHasName_LongNameIsNameWithBitPackedPrepended(self):
        fake_bool_data = np.full((5, 2), False)
        fake_array = xr.DataArray(
            fake_bool_data,
            dims=["y", "flag"],
            coords={"flag": ["flag1", "flag2"]},
            name="Test Flags",
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        self.assertEqual(result.attrs["long_name"], "Bit-packed Test Flags")

    def test_InputArrayHasNameAndLongNameAttr_UseLongName(self):
        fake_bool_data = np.full((5, 2), False)
        fake_array = xr.DataArray(
            fake_bool_data,
            dims=["y", "flag"],
            coords={"flag": ["flag1", "flag2"]},
            name="Fake Test Flags",
            attrs={"long_name": "Test Flags"},
        )
        result = fake_array.bitpacker.packbits(dim="flag")
        self.assertEqual(result.attrs["long_name"], "Bit-packed Test Flags")

    def test_NoCoordsForDim_UseDefaultFlagNames(self):
        fake_bool_data = np.full((2, 5), False)
        fake_array = xr.DataArray(fake_bool_data, dims=["flag", "y"])
        result = fake_array.bitpacker.packbits(dim="flag")
        self.assertEqual(result.attrs["bit_flags"], "flag_0 | flag_1")


class Test_unpackbits(unittest.TestCase):
    def test_AllZeros_UnpackAllFalse(self):
        fake_packed = xr.DataArray(
            np.zeros((5,), dtype=np.uint8),
            dims=["y"],
            attrs={
                "long_name": "Bit-packed flags",
                "bitorder": "little",
                "bit_flags": "flag_1 | flag_2",
                "bit_flag_separator": " | ",
            },
        )
        result = fake_packed.bitpacker.unpackbits(dim="flag")
        mock_result = xr.DataArray(
            np.full((5, 2), False),
            dims=["y", "flag"],
            coords={"flag": ["flag_1", "flag_2"]},
        )
        xr.testing.assert_equal(result, mock_result)

    def test_AllOnesBitorderLittle_UnpackFlag1True(self):
        fake_packed = xr.DataArray(
            np.ones((5,), dtype=np.uint8),
            dims=["y"],
            attrs={
                "long_name": "Bit-packed flags",
                "bitorder": "little",
                "bit_flags": "flag_1 | flag_2",
                "bit_flag_separator": " | ",
            },
        )
        result = fake_packed.bitpacker.unpackbits(dim="flag")
        mock_result = xr.DataArray(
            np.stack([np.full((5,), True), np.full((5,), False)], axis=-1),
            dims=["y", "flag"],
            coords={"flag": ["flag_1", "flag_2"]},
        )
        xr.testing.assert_equal(result, mock_result)

    def test_AllOnesBitorderLittleCommaSeparator_UnpackFlag1True(self):
        fake_packed = xr.DataArray(
            np.ones((5,), dtype=np.uint8),
            dims=["y"],
            attrs={
                "long_name": "Bit-packed flags",
                "bitorder": "little",
                "bit_flags": "flag_1, flag_2",
                "bit_flag_separator": ", ",
            },
        )
        result = fake_packed.bitpacker.unpackbits(dim="flag")
        mock_result = xr.DataArray(
            np.stack([np.full((5,), True), np.full((5,), False)], axis=-1),
            dims=["y", "flag"],
            coords={"flag": ["flag_1", "flag_2"]},
        )
        xr.testing.assert_equal(result, mock_result)

    def test_AllThreesBitorderLittle_UnpackAllFlagsTrue(self):
        fake_packed = xr.DataArray(
            3 * np.ones((5,), dtype=np.uint8),
            dims=["y"],
            attrs={
                "long_name": "Bit-packed flags",
                "bitorder": "little",
                "bit_flags": "flag_1 | flag_2",
                "bit_flag_separator": " | ",
            },
        )
        result = fake_packed.bitpacker.unpackbits(dim="flag")
        mock_result = xr.DataArray(
            np.full((5, 2), True),
            dims=["y", "flag"],
            coords={"flag": ["flag_1", "flag_2"]},
        )
        xr.testing.assert_equal(result, mock_result)

    def test_All128BitorderBig_UnpackFlag1True(self):
        fake_packed = xr.DataArray(
            128 * np.ones((5,), dtype=np.uint8),
            dims=["y"],
            attrs={
                "long_name": "Bit-packed flags",
                "bitorder": "big",
                "bit_flags": "flag_1 | flag_2",
                "bit_flag_separator": " | ",
            },
        )
        result = fake_packed.bitpacker.unpackbits(dim="flag")
        mock_result = xr.DataArray(
            np.stack([np.full((5,), True), np.full((5,), False)], axis=-1),
            dims=["y", "flag"],
            coords={"flag": ["flag_1", "flag_2"]},
        )
        xr.testing.assert_equal(result, mock_result)

    def test_Axis0_UnpackAlongAxis0(self):
        fake_packed = xr.DataArray(
            np.zeros((5,), dtype=np.uint8),
            dims=["y"],
            attrs={
                "long_name": "Bit-packed flags",
                "bitorder": "little",
                "bit_flags": "flag_1 | flag_2",
                "bit_flag_separator": " | ",
            },
        )
        result = fake_packed.bitpacker.unpackbits(dim="flag", axis=0)
        mock_result = xr.DataArray(
            np.full((2, 5), False),
            dims=["flag", "y"],
            coords={"flag": ["flag_1", "flag_2"]},
        )
        xr.testing.assert_equal(result, mock_result)

    def test_BitorderAndBitFlagsAndSeparatorNotInAttrs(self):
        fake_packed = xr.DataArray(
            np.zeros((5,), dtype=np.uint8),
            dims=["y"],
            attrs={
                "long_name": "Bit-packed flags",
                "bitorder": "little",
                "bit_flags": "flag_1 | flag_2",
                "bit_flag_separator": " | ",
            },
        )
        result = fake_packed.bitpacker.unpackbits(dim="flag", axis=0)
        with self.subTest("bitorder"):
            self.assertNotIn("bitorder", result.attrs)
        with self.subTest("bit_flags"):
            self.assertNotIn("bit_flags", result.attrs)
        with self.subTest("separator"):
            self.assertNotIn("bit_flag_separator", result.attrs)
