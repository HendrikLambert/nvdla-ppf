import unittest
import argparse

from model_helper import (
    export_onnx_model,
    build_loadable_from_onnx,
    generate_benchmark_onnx,
)


def subroutine_build_onnx():
    """
    Subroutine to build ONNX model for PFB FFT or DFT based on command line arguments.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Build ONNX model for PFB FFT or DFT.")

    # Argument to specify the type of model to export
    parser.add_argument(
        "--model_type",
        "-m",
        type=str,
        choices=["pfb_fft", "pfb_dft"],
        required=True,
        help="Type of model to export: 'pfb_fft' for PFB FFT or 'pfb_dft' for PFB DFT.",
    )

    # Batch size argument
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Batch size for the model export.",
    )
    # Taps argument
    parser.add_argument(
        "--taps",
        "-t",
        type=int,
        default=16,
        help="Number of taps for the model export.",
    )
    # Amount of channels argument
    parser.add_argument(
        "--channels",
        "-c",
        type=int,
        default=256,
        help="Number of channels for the model export.",
    )

    # File name argument
    parser.add_argument(
        "--onnx_file",
        "-o",
        type=str,
        required=True,
        help="Location of the ONNX file to export to.",
    )

    args, _ = parser.parse_known_args()

    export_onnx_model(
        b=args.batch_size,
        c=args.channels,
        t=args.taps,
        model_type=args.model_type,
        file=args.onnx_file,
    )


def subroutine_build_nvdla():
    """
    Subroutine to build NVDLA loadable from ONNX file.
    """
    parser = argparse.ArgumentParser(
        description="Build the NVDLA loadable from ONNX file."
    )

    # Input ONNX file argument
    parser.add_argument(
        "--onnx_file",
        "-o",
        type=str,
        required=True,
        help="Location of the ONNX file to import from.",
    )

    # Export location
    parser.add_argument(
        "--loadable_location",
        "-l",
        type=str,
        required=True,
        help="Location to export the NVDLA loadable.",
    )

    # TRTEXEC location
    parser.add_argument(
        "--trtexec_location",
        "-t",
        type=str,
        default="/usr/src/tensorrt/bin/trtexec",
        help="Location of the trtexec binary.",
    )

    # Verbose flag
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output."
    )

    args, _ = parser.parse_known_args()
    # Call the function to build the NVDLA loadable
    build_loadable_from_onnx(
        onnx_file=args.onnx_file,
        loadable_location=args.loadable_location,
        trtexe_location=args.trtexec_location,
        verbose=args.verbose,
    )


def subroutine_build_benchmark():
    """
    Subroutine to build benchmark files.
    """
    parser = argparse.ArgumentParser(description="Build the benchmark files.")
    # Args for bulding ONNX or NVDLA loadable
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        choices=["onnx", "nvdla"],
        required=True,
        help="Type of benchmark to build: 'onnx' for ONNX model or 'nvdla' for NVDLA loadable.",
    )

    # ONNX folder
    parser.add_argument(
        "--onnx_folder",
        "-o",
        type=str,
        required=True,
        help="Folder to save the ONNX models.",
    )
    # NVDLA folder
    parser.add_argument(
        "--nvdla_folder", "-n", type=str, help="Folder to save the NVDLA loadables."
    )

    args, _ = parser.parse_known_args()

    # Switch between ONNX and NVDLA loadable generation
    if args.type == "onnx":
        generate_benchmark_onnx(args.onnx_folder)


def main():
    """ "
    Main function to run the export functions based on command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="A tool to build and export PFB models as ONNX or NVDLA loadables."
    )

    # Switch between main modes
    parser.add_argument(
        "--buildONNX", action="store_true", help="Export PFB FFT or DFT model."
    )
    parser.add_argument(
        "--buildNVDLA",
        action="store_true",
        help="Export NVDLA loadable from ONNX file.",
    )
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument(
        "--benchmark", action="store_true", help="Build benchmark files."
    )

    args, _ = parser.parse_known_args()

    # Run the appropriate subroutine based on the arguments
    if args.test:
        # Run the unittests
        print("Running tests...")
        # Discover the tests
        suite = unittest.TestLoader().discover("tests", pattern="test*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)

    if args.benchmark:
        subroutine_build_benchmark()
        return  # Dont run the rest of the subroutines

    if args.buildONNX:
        subroutine_build_onnx()
    if args.buildNVDLA:
        subroutine_build_nvdla()


if __name__ == "__main__":
    main()
