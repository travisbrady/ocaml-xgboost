open Printf
open Ctypes
open Foreign

let from = Dl.(dlopen ~filename:"libxgboostwrapper.so" ~flags:[RTLD_NOW])
let get_last_error = foreign ~from "XGBGetLastError" (void @-> returning string)

module Matrix = struct
    let t = ptr void

    let _create_from_csr = foreign ~from
        "XGDMatrixCreateFromCSR"
        (ptr ulong @-> ptr uint @-> ptr float @-> ulong @-> ulong @-> ptr t @-> returning int)
    let _from_mat = foreign ~from
        "XGDMatrixCreateFromMat" (ptr float @->
                                  ulong @-> ulong @->
                                  float @-> ptr t @-> returning int)
    let of_list lst nrows ncols =
        let mat = CArray.of_list float lst in
        let nr = Unsigned.ULong.of_int nrows in
        let nc = Unsigned.ULong.of_int ncols in
        let miss = -999.0 in
        let h = allocate_n ~count:1 (ptr void) in
        let ret = _from_mat (CArray.start mat) nr nr miss h in
        !@h

    let from_mat () =
        let mat = CArray.of_list float [0.0; 1.0; 2.0; 3.0] in
        let nr = Unsigned.ULong.of_int 2 in
        let miss = -999.0 in
        let h = allocate_n ~count:1 (ptr void) in
        let ret = _from_mat (CArray.start mat) nr nr miss h in
        printf "[from_mat] RET: %d\n" ret;
        ret, !@h

    let _from_file = foreign ~from "XGDMatrixCreateFromFile" (string @->
        int @-> ptr t @-> returning int)

    let from_file fn =
        let h = allocate_n ~count:1 (ptr void) in
        let ret = _from_file fn 0 h in
        ret, !@h

    let _num_rows = foreign ~from "XGDMatrixNumRow" (t @-> ptr ulong @-> returning int)

    let num_rows m = 
        let ul = Unsigned.ULong.of_int 99 in
        let out = allocate ulong ul in
        let ret = _num_rows m out in
        ret, Unsigned.ULong.to_int !@out, Unsigned.ULong.to_string !@out

    let _save_binary = foreign ~from "XGDMatrixSaveBinary" (t @-> string @-> int @-> returning int)

    let save_binary m fn = 
        let ret = _save_binary m fn 1 in
        ret

    let _set_group = foreign ~from "XGDMatrixSetGroup" (t @-> ptr uint @-> ulong @-> returning int)

    let set_group m labels =
        let ulab = List.map (fun x -> Unsigned.UInt.of_int x) labels in
        let laba = CArray.of_list uint ulab in
        let len = Unsigned.ULong.of_int (List.length labels) in
        _set_group m (CArray.start laba) len

    let _set_float_info = foreign ~from "XGDMatrixSetFloatInfo"
        (t @-> string @-> ptr float @-> ulong @-> returning int)

    let set_label m labels =
        printf "[set_label] INSIDE\n%!";
        let arr = CArray.start (CArray.of_list float labels) in
        let len = Unsigned.ULong.of_int (List.length labels) in
        printf "[set_label] _set_float_info\n%!";
        let ret = _set_float_info m "label" arr len in
        printf "[set_label] %d\n%!" ret;
        ret

    let free = foreign ~from "XGDMatrixFree" (t @-> returning int)

end

module Booster = struct
    type t
    let t = ptr void
    let _create = foreign ~from "XGBoosterCreate" 
        (ptr (ptr void) @->
         ulong @->
         ptr t @->
         returning int)

    let create () =
        let dmats = allocate_n ~count:1 (ptr void) in
        let len = Unsigned.ULong.of_int 0 in
        let h = allocate_n ~count:1 (ptr void) in
        let ret = _create dmats len h in
        !@h

    let create2 dtrain =
        let dmats = CArray.of_list Matrix.t [dtrain] in
        let len = Unsigned.ULong.of_int 1 in
        let h = allocate_n ~count:1 (ptr void) in
        let ret = _create (CArray.start dmats) len h in
        !@h

    let free = foreign ~from "XGBoosterFree" (t @-> returning int)

    let _set_param = foreign ~from 
        "XGBoosterSetParam" 
        (t @-> string @-> string @-> returning int)
    let set_param booster name value =
        let ret = _set_param booster name value in
        printf "[set_param] %s %s ret: %d\n" name value ret

    let _update_one_iter = foreign ~from ~release_runtime_lock:true
        ~check_errno:true
        "XGBoosterUpdateOneIter"
        (t @-> int @-> Matrix.t @-> returning int)

    let _boost_one_iter = foreign ~from
        "XGBoosterBoostOneIter"
        (t @-> Matrix.t @-> ptr float @-> ptr float @-> ulong @-> returning int)

    let _eval_one_iter = foreign ~from ~release_runtime_lock:true
        "XGBoosterEvalOneIter"
        (t @-> int @-> ptr Matrix.t @->
            ptr string @-> ulong @-> ptr string @-> returning int)

    let _predict = foreign ~from
        "XGBoosterPredict"
        (t @-> Matrix.t @-> int @-> uint @-> ptr ulong @-> ptr (ptr float) @-> returning int)

    let _save_model = foreign ~from "XGBoosterSaveModel" (t @-> string @-> returning int)

    let _load_model_from_buffer = foreign ~from 
        "XGBoosterLoadModelFromBuffer"
        (t @-> ptr void @-> ulong @-> returning int)

    let _get_model_raw = foreign ~from
        "XGBoosterGetModelRaw"
        (t @-> ptr ulong @-> ptr string @-> returning int)

    let _dump_model = foreign ~from
        "XGBoosterDumpModel"
        (t @-> string @-> int @-> ptr ulong @-> ptr (ptr string) @-> returning int)

    let train b dtrain num_rounds =
        let dmats = CArray.start (CArray.of_list Matrix.t [dtrain]) in
        let enames = CArray.start (CArray.of_list string ["dtrain"]) in
        let evres = allocate_n ~count:1024 string in
        let len = Unsigned.ULong.of_int 1 in
        for i = 0 to num_rounds-1 do
            printf "[train] %d\n%!" i;
            let ret = _update_one_iter b i dtrain in
            printf "[_update_one_iter] %d\n%!" ret;
            let ret = _eval_one_iter b i dmats enames len evres in
            printf "[train] ret: %d\n %s\n%!" ret !@evres;
            ()
        done
end

module XGBoostClassifier = struct
end
