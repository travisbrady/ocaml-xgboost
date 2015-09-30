let printf = Printf.printf

let () =
    let _, dtrain = Oxgboost.Matrix.from_file "agaricus.txt.train" in
    let _, dtest = Oxgboost.Matrix.from_file "agaricus.txt.test" in
    let bst = Oxgboost.XGBEstimator.fit ~objective:`Binary_logistic ~n_estimators:2 dtrain in
    let y_true = Oxgboost.Matrix.get_float_info dtest "label" in
    let _y_pred = Oxgboost.Booster.predict bst dtest in
    let y_pred = Array.to_list _y_pred in

    let y_true_i = List.map int_of_float y_true in
    let y_pred_i = List.map (fun x -> int_of_float (x +. 0.5)) y_pred in

    let cm = Util.confusion_matrix y_true_i y_pred_i in

    printf "confusion matrix\n";
    Hashtbl.iter (fun (k1, k2) v ->
        printf "%d %d %6d\n" k1 k2 v
    ) cm;
    let acc = Util.accuracy_score y_true_i y_pred_i in
    printf "Acc: %.4f\n" acc;
    printf "Prec: %.4f\n" (Util.precision_score y_true_i y_pred_i);
    printf "Recall: %.4f\n" (Util.recall_score y_true_i y_pred_i);
    ()
