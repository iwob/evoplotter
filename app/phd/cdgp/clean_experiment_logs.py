from src.templates import *
from src.utils import *
import app.phd.cdgp.phd_processor as phd_processor
import app.phd.cdgp.phd_utils as phd_utils



def cleanLIA():
    folders_in = ["LIA_ORIG"]
    folder_out = "LIA/"
    ensure_clear_dir(folder_out)

    props = phd_utils.load_correct_props_simple(folders_in)
    phd_processor.standardize_benchmark_names(props)
    dim_benchmark = phd_processor.get_benchmarks_from_props(props)
    dim_method = phd_processor.dim_methodCDGP + phd_processor.dim_methodGPR
    dim = phd_processor.dim_evoMode * dim_benchmark * dim_method * phd_processor.dim_sel * phd_processor.dim_testsRatio
    utils.reorganizeExperimentFiles(props, dim, target_dir=folder_out, maxRuns=50)



def cleanSLIA():
    folders_in = ["SLIA_ORIG"]
    folder_out = "SLIA/"
    ensure_clear_dir(folder_out)

    props = phd_utils.load_correct_props_simple(folders_in)
    phd_processor.standardize_benchmark_names(props)
    dim_benchmark = phd_processor.get_benchmarks_from_props(props)
    dim_method = phd_processor.dim_methodCDGP
    dim = phd_processor.dim_evoMode * dim_benchmark * dim_method * phd_processor.dim_sel * phd_processor.dim_testsRatio
    utils.reorganizeExperimentFiles(props, dim, target_dir=folder_out, maxRuns=50)



if __name__ == "__main__":
    cleanLIA()
    # cleanSLIA()
