open Ctypes
open Foreign

let from = Dl.(dlopen ~filename:"libxgboostwrapper.so" ~flags:[RTLD_NOW])
let get_last_error = foreign ~from "XGBGetLastError" (void @-> returning string)

module DMatrix = struct
    let t = ptr void

    let create_from_csr = foreign ~from
        "XGDMatrixCreateFromCSR"
        (ptr ulong @-> ptr uint @-> ptr float @-> ulong @-> ulong @-> ptr t @-> returning int)

    let create_from_mat = foreign ~from
        "XGDMatrixCreateFromMat" (ptr float @->
                                  ulong @-> ulong @->
                                  float @-> ptr t @-> returning int)

    let create_from_file = foreign ~from "XGDMatrixCreateFromFile" (string @->
        int @-> ptr t @-> returning int)

    let num_row = foreign ~from "XGDMatrixNumRow" (t @-> ptr ulong @-> returning int)

    let num_col = foreign ~from "XGDMatrixNumCol" (t @-> ptr ulong @-> returning int)

    let save_binary = foreign ~from "XGDMatrixSaveBinary" (t @-> string @-> int @-> returning int)

    let set_group = foreign ~from "XGDMatrixSetGroup" (t @-> ptr uint @-> ulong @-> returning int)

    let set_float_info = foreign ~from "XGDMatrixSetFloatInfo"
        (t @-> string @-> ptr float @-> ulong @-> returning int)

    let get_float_info =
        foreign ~from "XGDMatrixGetFloatInfo"
        (t @-> string @-> ptr ulong @-> ptr (ptr float) @-> returning int)

    let get_uint_info = foreign ~from "XGDMatrixGetUIntInfo"
        (t @-> string @-> ptr ulong @-> ptr (ptr ulong) @-> returning int)

    let free = foreign ~from "XGDMatrixFree" (t @-> returning int)

end

module Booster = struct
    type t
    let t = ptr void

    let create = foreign ~from "XGBoosterCreate" 
        (ptr (ptr void) @->
         ulong @->
         ptr t @->
         returning int)

    let free = foreign ~from "XGBoosterFree" (t @-> returning int)

    let set_param = foreign ~from 
        "XGBoosterSetParam" 
        (t @-> string @-> string @-> returning int)

    let update_one_iter = foreign ~from ~release_runtime_lock:true
        ~check_errno:true
        "XGBoosterUpdateOneIter"
        (t @-> int @-> DMatrix.t @-> returning int)

    let boost_one_iter = foreign ~from
        "XGBoosterBoostOneIter"
        (t @-> DMatrix.t @-> ptr float @-> ptr float @-> ulong @-> returning int)

    let eval_one_iter = foreign ~from ~release_runtime_lock:true
        "XGBoosterEvalOneIter"
        (t @-> int @-> ptr DMatrix.t @->
            ptr string @-> ulong @-> ptr string @-> returning int)

    let predict = foreign ~from
        "XGBoosterPredict"
        (t @-> DMatrix.t @-> int @-> uint @-> ptr ulong @-> ptr (ptr float) @-> returning int)

    let save_model = foreign ~from "XGBoosterSaveModel" (t @-> string @-> returning int)

    let load_model_from_buffer = foreign ~from 
        "XGBoosterLoadModelFromBuffer"
        (t @-> ptr void @-> ulong @-> returning int)

    let get_model_raw = foreign ~from
        "XGBoosterGetModelRaw"
        (t @-> ptr ulong @-> ptr string @-> returning int)

    let dump_model = foreign ~from
        "XGBoosterDumpModel"
        (t @-> string @-> int @-> ptr ulong @-> ptr (ptr string) @-> returning int)

    let dump_model_with_features = foreign ~from
        "XGBoosterDumpModelWithFeatures"
        (t @-> int @-> ptr string @-> ptr string @-> int @-> ptr ulong @-> ptr string @-> returning int)
end
