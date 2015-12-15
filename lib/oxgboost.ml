open Printf
open Ctypes
open Foreign
module X=Xgboost

let from = Dl.(dlopen ~filename:"libxgboostwrapper.so" ~flags:[RTLD_NOW])
let get_last_error = foreign ~from "XGBGetLastError" (void @-> returning string)

module Matrix = struct
    let t = ptr void

    let of_bigarray ?(missing=(-999.0)) ba =
        let nrows = Bigarray.Array2.dim1 ba in
        let ncols = Bigarray.Array2.dim2 ba in
        let nr = Unsigned.ULong.of_int nrows in
        let nc = Unsigned.ULong.of_int ncols in
        let h = allocate_n ~count:1 (ptr void) in
        let ret = X.DMatrix.create_from_mat (bigarray_start array2 ba) nr nc missing h in
        !@h

    let of_list lst nrows ncols =
        let mat = CArray.of_list float lst in
        let nr = Unsigned.ULong.of_int nrows in
        let nc = Unsigned.ULong.of_int ncols in
        let miss = -999.0 in
        let h = allocate_n ~count:1 (ptr void) in
        let ret = X.DMatrix.create_from_mat (CArray.start mat) nr nc miss h in
        !@h

    let from_mat () =
        let mat = CArray.of_list float [0.0; 1.0; 2.0; 3.0] in
        let nr = Unsigned.ULong.of_int 2 in
        let miss = -999.0 in
        let h = allocate_n ~count:1 (ptr void) in
        let ret = X.DMatrix.create_from_mat (CArray.start mat) nr nr miss h in
        ret, !@h

    let from_file fn =
        let h = allocate_n ~count:1 (ptr void) in
        let ret = X.DMatrix.create_from_file fn 0 h in
        ret, !@h

    let num_rows m = 
        let ul = Unsigned.ULong.of_int 99 in
        let out = allocate ulong ul in
        let ret = X.DMatrix.num_row m out in
        ret, Unsigned.ULong.to_int !@out, Unsigned.ULong.to_string !@out

    let num_cols m =
        let ul = Unsigned.ULong.of_int 99 in
        let out = allocate ulong ul in
        let ret = X.DMatrix.num_row m out in
        Unsigned.ULong.to_int !@out

    let save_binary m fn = 
        let ret = X.DMatrix.save_binary m fn 1 in
        ret

    let set_group m labels =
        let ulab = List.map (fun x -> Unsigned.UInt.of_int x) labels in
        let laba = CArray.of_list uint ulab in
        let len = Unsigned.ULong.of_int (List.length labels) in
        X.DMatrix.set_group m (CArray.start laba) len

    let set_label m labels =
        let arr = CArray.start (CArray.of_list float labels) in
        let len = Unsigned.ULong.of_int (List.length labels) in
        let ret = X.DMatrix.set_float_info m "label" arr len in
        ret

    let get_float_info m field =
        let ul = Unsigned.ULong.of_int 0 in
        let out_ul = allocate ulong ul in
        let out_result = allocate (ptr float) (allocate float 0.0) in
        let rc = X.DMatrix.get_float_info m field out_ul out_result in
        let out_len = Unsigned.ULong.to_int !@out_ul in
        CArray.to_list (CArray.from_ptr !@out_result out_len)

    let get_uint_info m field =
        let oi = Unsigned.ULong.of_int in
        let ul = oi 0 in
        let out_ul = allocate ulong ul in
        let out_result = allocate (ptr ulong) (allocate ulong (oi 0)) in
        let rc = X.DMatrix.get_uint_info m field out_ul out_result in
        let out_len = Unsigned.ULong.to_int !@out_ul in
        CArray.to_list (CArray.from_ptr !@out_result out_len)

    let free m = X.DMatrix.free m

end

module Booster = struct
    type t
    let t = ptr void

    let string_of_booster_type = function
        | `GBTree -> "gbtree"
        | `Gblinear -> "gblinear"

    let string_of_objective = function
        | `Reg_linear -> "reg:linear"
        | `Reg_logistic -> "reg:logistic"
        | `Binary_logistic -> "binary:logistic"
        | `Binary_logitraw -> "binary:logitraw"
        | `Count_poisson -> "count:poisson"
        | `Multi_softmax -> "multi:softmax"
        | `Multi_softprob -> "multi:softprob"
        | `Rank_pairwise -> "rank:pairwise"

    let _create = foreign ~from "XGBoosterCreate" 
        (ptr (ptr void) @->
         ulong @->
         ptr t @->
         returning int)

    let create () =
        let dmats = allocate_n ~count:1 (ptr void) in
        let len = Unsigned.ULong.of_int 0 in
        let h = allocate_n ~count:1 (ptr void) in
        let ret = X.Booster.create dmats len h in
        !@h

    let create2 dtrain =
        let dmats = CArray.of_list Matrix.t [dtrain] in
        let len = Unsigned.ULong.of_int 1 in
        let h = allocate_n ~count:1 (ptr void) in
        let ret = X.Booster.create (CArray.start dmats) len h in
        !@h

    let free bst = X.Booster.free bst

    let set_param booster name value =
        let ret = X.Booster.set_param booster name value in
        printf "[set_param] %s %s ret: %d\n" name value ret

    let predict ?(option_mask=0) ?(ntree_limit=0) bst mat =
        let u_ntree_limit = Unsigned.UInt.of_int ntree_limit in
        let _, y_test_len, _ = Matrix.num_rows mat in
        let ul = Unsigned.ULong.of_int y_test_len in
        let u_out_len = allocate ulong ul in
        let inner_out = allocate_n ~count:y_test_len float in
        for i = 0 to y_test_len-1 do
            (inner_out +@ i) <-@ 0.0
        done;
        let out_result = allocate (ptr float) inner_out in
        let ret = X.Booster.predict bst mat option_mask u_ntree_limit u_out_len out_result in
        let out_len = Unsigned.ULong.to_int !@u_out_len in
        let arr = Array.make out_len 0.0 in
        let rezzo = !@out_result in
        for i = 0 to out_len-1 do
            let res = !@(rezzo +@ i) in
            arr.(i) <- res
        done;
        arr

    let train b dtrain num_rounds =
        let dmats = CArray.start (CArray.of_list Matrix.t [dtrain]) in
        let enames = CArray.start (CArray.of_list string ["dtrain"]) in
        let evres = allocate_n ~count:1024 string in
        let len = Unsigned.ULong.of_int 1 in
        for i = 0 to num_rounds-1 do
            printf "[train] %d\n%!" i;
            let ret = X.Booster.update_one_iter b i dtrain in
            let ret = X.Booster.eval_one_iter b i dmats enames len evres in
            printf "[train] ret: %d\n %s\n%!" ret !@evres;
            ()
        done
end

module XGBRegressor = struct
    type t
    let fit ?(n_estimators=100) ?(seed=0) ?(booster=`GBTree) ?(max_depth=3) dtrain =
        let bst = Booster.create2 dtrain in
        Booster.set_param bst "seed" (string_of_int seed);
        Booster.set_param bst "booster" (Booster.string_of_booster_type booster);
        Booster.set_param bst "objective" "reg:linear";
        Booster.train bst dtrain n_estimators;
        bst

end

module XGBEstimator = struct
    type t
    let fit ?(booster=`GBTree) ?(n_estimators=100) ?(max_depth=3) ?(learning_rate=0.1)
        ?(objective=`Binary_logistic) ?(gamma=0.0) ?(min_child_weight=1.0)
        ?(max_delta_step=0) ?(subsample=1.0) ?(colsample_bytree=1)
        ?(base_score=0.5) ?(seed=0) dtrain =
        let bst = Booster.create2 dtrain in
        Booster.set_param bst "booster" (Booster.string_of_booster_type booster);
        Booster.set_param bst "max_depth" (string_of_int max_depth);
        Booster.set_param bst "eta" (string_of_float learning_rate);
        Booster.set_param bst "objective" (Booster.string_of_objective objective);
        Booster.set_param bst "gamma" (string_of_float gamma);
        Booster.set_param bst "base_score" (string_of_float base_score);
        Booster.set_param bst "seed" (string_of_int seed);
        Booster.train bst dtrain n_estimators;
        bst
end


