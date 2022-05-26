import os
import time
import pathlib
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=pathlib.Path, required=True)

parser.add_argument(
    "--outputpath", type=pathlib.Path, default=pathlib.Path("output.json")
)
parser.add_argument("--modelweights", type=pathlib.Path)

parser.add_argument("--num_beams", type=int, default=32)
parser.add_argument("--ngram_size_penalty", type=int, default=2)
parser.add_argument("--n_repetitions", type=int, default=1)
parser.add_argument("--frequency_known", action="store_true")

args = parser.parse_args()


beamsearch_rust_dir = pathlib.Path(__file__).parent / "beamsearch_rust"
cargo_path = subprocess.run(["which", "cargo"], stdout=subprocess.PIPE).stdout.decode(
    "utf-8"
)[:-1]
process_env = os.environ.copy()

print("Running beamsearch with the following args", args)
process = subprocess.Popen(
    [
        f"{cargo_path} run "
        "--release "
        "--manifest-path "
        f"{(beamsearch_rust_dir / 'Cargo.toml').absolute().as_posix()} "
        "-- "
        "--num-beams "
        f"{args.num_beams} "
        "--datapath "
        f"{args.datapath.absolute().as_posix()} "
        "--outputpath "
        f"{args.outputpath} "
        "--n-repetitions "
        f"{args.n_repetitions} "
        f"{'--no-frequencies' if not args.frequency_known else ''} "
        f"--no-repeat-ngram-size "
        f"{args.ngram_size_penalty} "
        f"{('--modelweights ' + str(args.modelweights)) if args.modelweights else ''} "
    ],
    shell=True,
    stdout=pathlib.Path("stdout.log").open("w"),
    stderr=pathlib.Path("stderr.log").open("w"),
    env=process_env,
)

while True:
    time.sleep(1)
    if process.poll() is not None:
        break
