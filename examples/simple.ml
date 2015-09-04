open Printf

let le = Oxgboost.get_last_error
let () =
    let lr, mat = Oxgboost.Matrix.from_file "agaricus.txt.test" in
    printf "load ret: %d %s\n" lr (le ());
    let ret, nr, x = Oxgboost.Matrix.num_rows mat in
    printf "Ret: %d NumRows: %d X: %s\n" ret nr x;
    let b = Oxgboost.Booster.create2 mat in
    (*
    let ret = Oxgboost.Booster._boost_update_one_iter b 1 mat in
    printf "_boost_update_one_iter %d\n" ret;
    *)
    printf "now set_param\n%!";
    Oxgboost.Booster.set_param b "seed" "0";
    Oxgboost.Booster.set_param b "booster" "gbtree";
    Oxgboost.Booster.set_param b "objective" "binary:logistic";
    Oxgboost.Booster.set_param b "eta" "0.1";
    Oxgboost.Booster.set_param b "gamma" "1.0";
    Oxgboost.Booster.set_param b "max_depth" "2";

    printf "now train\n%!";
    Oxgboost.Booster.train b mat 20;
    printf "training done\n%!";

    let sr = Oxgboost.Matrix.save_binary mat "machine.mat" in
    printf "SaveRes: %d %s\n" sr (Oxgboost.get_last_error ());
    printf "Last Error: %s\n" (Oxgboost.get_last_error ());

    let ret, mat = Oxgboost.Matrix.from_mat () in
    printf "Made Mat: %d LE: %s\n" ret (le());
    let _ = Oxgboost.Matrix.set_label mat [1.0; 0.0] in
    let ret, nr, _ = Oxgboost.Matrix.num_rows mat in
    printf "FromMat NumRows: %d Ret: %d\n" nr ret;
    (*
    let free_res = Oxgboost.Matrix.free mat in
    printf "Free Res: %d\n" free_res;
    *)
    ()

