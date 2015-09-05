open Printf

let o_or_z x = if x = 1.0 then 1.0 else 0.0

let () =
    let x = [
        [0.0; 0.0; 0.0];
        [0.0; 0.0; 0.0];
        [1.0; 0.0; 0.0];
        [1.0; 0.0; 0.0];
        [0.0; 1.0; 0.0];
        [0.0; 1.0; 0.0];
        [0.0; 0.0; 1.0];
        [0.0; 0.0; 1.0];
        [1.0; 1.0; 0.0];
        [1.0; 1.0; 0.0];
        [1.0; 1.0; 1.0];
        [1.0; 1.0; 1.0];
        [0.0; 1.0; 1.0];
        [0.0; 1.0; 1.0];
        [1.0; 0.0; 1.0];
        [1.0; 0.0; 1.0];
        [0.0; 0.0; 0.0];
        [0.0; 0.0; 0.0];
        [1.0; 0.0; 0.0];
        [1.0; 0.0; 0.0];
        [0.0; 1.0; 0.0];
        [0.0; 1.0; 0.0];
        [0.0; 0.0; 1.0];
        [0.0; 0.0; 1.0];
        [1.0; 1.0; 0.0];
        [1.0; 1.0; 0.0];
        [1.0; 1.0; 1.0];
        [1.0; 1.0; 1.0];
        [0.0; 1.0; 1.0];
        [0.0; 1.0; 1.0];
        [1.0; 0.0; 1.0];
        [1.0; 0.0; 1.0];
        [0.0; 0.0; 0.0];
        [0.0; 0.0; 0.0];
        [1.0; 0.0; 0.0];
        [1.0; 0.0; 0.0];
        [0.0; 1.0; 0.0];
        [0.0; 1.0; 0.0];
        [0.0; 0.0; 1.0];
        [0.0; 0.0; 1.0];
        [1.0; 1.0; 0.0];
        [1.0; 1.0; 0.0];
        [1.0; 1.0; 1.0];
        [1.0; 1.0; 1.0];
        [0.0; 1.0; 1.0];
        [0.0; 1.0; 1.0];
        [1.0; 0.0; 1.0];
        [1.0; 0.0; 1.0];
        [0.0; 0.0; 0.0];
        [0.0; 0.0; 0.0];
        [1.0; 0.0; 0.0];
        [1.0; 0.0; 0.0];
        [0.0; 1.0; 0.0];
        [0.0; 1.0; 0.0];
        [0.0; 0.0; 1.0];
        [0.0; 0.0; 1.0];
        [1.0; 1.0; 0.0];
        [1.0; 1.0; 0.0];
        [1.0; 1.0; 1.0];
        [1.0; 1.0; 1.0];
        [0.0; 1.0; 1.0];
        [0.0; 1.0; 1.0];
        [1.0; 0.0; 1.0];
        [1.0; 0.0; 1.0]
    ] in
    let y = List.map (fun row ->
        o_or_z (List.fold_left (fun a b -> a +. b) 0.0 row)
    ) x in
    let x_flat = List.flatten x in
    let mat = Oxgboost.Matrix.of_list x_flat (List.length x) 3 in

    let _, num_rows, _ = Oxgboost.Matrix.num_rows mat in
    printf "%d\n%!" num_rows;

    printf "Now set_label\n%!";
    let _ = Oxgboost.Matrix.set_label mat y in
    let b = Oxgboost.Booster.create2 mat in
    Oxgboost.Booster.set_param b "seed" "0";
    Oxgboost.Booster.set_param b "booster" "gbtree";
    Oxgboost.Booster.set_param b "objective" "binary:logistic";
    Oxgboost.Booster.set_param b "eta" "0.1";
    Oxgboost.Booster.set_param b "gamma" "1.0";
    Oxgboost.Booster.set_param b "max_depth" "7";
    Oxgboost.Booster.train b mat 175;
    printf "training done\n%!";
    let y_test = Oxgboost.Matrix.of_list [
        1.0; 0.0; 0.0;
        0.0; 0.0; 0.0;
        0.0; 0.0; 1.0;
        0.0; 1.0; 0.0;
        1.0; 1.0; 1.0] 5 3 in
    let y_pred = Oxgboost.Booster.predict b y_test in
    printf "YO!\n%!";
    Array.iter (fun x -> printf "%f " x) y_pred;
    ()
