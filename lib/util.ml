let confusion_matrix y_true y_pred =
    let h = Hashtbl.create 20 in
    List.iter2 (fun yt yp ->
        try
            let v = Hashtbl.find h (yt, yp) in
            Hashtbl.replace h (yt, yp) (v + 1)
        with Not_found ->
            Hashtbl.add h (yt, yp) 1
    ) y_true y_pred;
    h

let accuracy_score y_true y_pred =
    let total = float_of_int (List.length y_true) in
    let same_labels = List.fold_left2 (fun acc yt yp ->
        if yt = yp then acc + 1 else acc
    ) 0 y_true y_pred in
    (float_of_int same_labels) /. total

let precision_score y_true y_pred =
    let tp, fp = List.fold_left2 (fun (tps, fps) yt yp ->
        match (yt, yp) with
            | 1, 1 -> (tps + 1, fps)
            | _, 1 -> (tps, fps + 1)
            | _ -> (tps, fps)
    ) (0, 0) y_true y_pred in
    let tpf, fpf = float_of_int tp, float_of_int fp in
    tpf /. (tpf +. fpf)

let recall_score y_true y_pred =
    let tp, fn = List.fold_left2 (fun (tps, fns) yt yp ->
        match (yt, yp) with
            | 1, 1 -> (tps + 1, fns)
            | 1, _ -> (tps, fns + 1)
            | _ -> (tps, fns)
    ) (0, 0) y_true y_pred in
    let tpf, fnf = float_of_int tp, float_of_int fn in
    tpf /. (tpf +. fnf)
