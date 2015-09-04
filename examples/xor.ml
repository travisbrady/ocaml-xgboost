open Printf

let o_or_z x = if x > 0.0 then 1.0 else 0.0

let () =
    let x = [
        [0.0; 0.0; 0.0];
        [1.0; 0.0; 0.0];
        [0.0; 1.0; 0.0];
        [0.0; 0.0; 1.0];
        [1.0; 1.0; 0.0];
        [1.0; 1.0; 1.0];
        [0.0; 1.0; 1.0];
        [1.0; 0.0; 1.0]
    ] in
    let y = List.map (fun row ->
        o_or_z (List.fold_left (fun a b -> a +. b) 0.0 row)
    ) x in
    let x_flat = List.flatten x in
    printf "Matrix.of_list\n%!";
    let mat = Oxgboost.Matrix.of_list x_flat (List.length x) 3 in
    printf "Now set_label\n%!";
    let _ = Oxgboost.Matrix.set_label mat y in
    ()
