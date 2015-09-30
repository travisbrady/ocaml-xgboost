let printf = Printf.printf

let comp_errors comp_func y_true y_pred =
    let lenf = float_of_int (List.length y_true) in
    let err_sum = List.fold_left2 (fun acc _true pred -> acc +. (comp_func _true pred)) 0.0 y_true y_pred in
    err_sum /. lenf

let mse = comp_errors (fun _true pred -> (_true -. pred) ** 2.0)
let mae = comp_errors (fun _true pred -> abs_float (_true -. pred))

let () =
    let lr, dtrain = Oxgboost.Matrix.from_file "housing.train" in
    let _, dtest = Oxgboost.Matrix.from_file "housing.test" in
    (*
    let bst = Oxgboost.XGBRegressor.fit ~n_estimators:25 dtrain in
    *)
    let bst = Oxgboost.XGBEstimator.fit ~objective:`Reg_linear ~n_estimators:25 dtrain in
    let y_true = Oxgboost.Matrix.get_float_info dtest "label" in
    let _y_pred = Oxgboost.Booster.predict bst dtest in
    let y_pred = Array.to_list _y_pred in
    let _mse = mse y_true y_pred in
    printf "MSE: %.3f\n" _mse;
    printf "RMSE: %.3f\n" (sqrt _mse);
    printf "MAE: %.3f\n" (mae y_true y_pred);
    List.iter2 (fun yt yp -> printf "%.1f %.1f\n" yt yp) y_true y_pred
