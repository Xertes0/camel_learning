open Camel_learning

module Mat = Matrix

let main () =
  let tx = Mat.from_3d_list
                  [ [ 0.; 0. ];
                    [ 0.; 1. ];
                    [ 1.; 0. ];
                    [ 1.; 1. ] ] in
  let ty = Mat.from_3d_list
                  [ [ 0. ];
                    [ 1. ];
                    [ 1. ];
                    [ 1. ] ] in

  let shape = [| 2; 2; 1; |] in
  let model = ref (Model.random shape) in

  for _ = 0 to 100_000 do
    (* let grad = Model.finite_difference !model ~tx ~ty ~eps:0.1 ~rate:0.1 in *)
    let grad = Model.back_propagation !model ~tx ~ty ~rate:1. in
    model := Model.apply_gradient !model grad;
    Printf.printf "Cost: %f\n" (Model.cost !model ~tx ~ty);
  done;

  Model.evaluate !model ~tx ~ty

;;
main ()
