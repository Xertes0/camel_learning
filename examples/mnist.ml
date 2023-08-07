open Camel_learning

module Mat = Matrix

let read_bin_file (path: string) =
  let chan = Stdlib.open_in_bin path in
  let chan_len = Stdlib.in_channel_length chan in
  let buf = Bytes.create chan_len in
  Stdlib.really_input chan buf 0 chan_len;
  Stdlib.close_in chan;
  buf

let labels_from_path (path: string): Mat.t =
  let buf = read_bin_file path in
  let count = Int32.to_int (Bytes.get_int32_be buf 4) in
  Mat.init count 10
    (fun row col ->
      if col = Bytes.get_uint8 buf (8 + row)
      then 1. else 0.)

let images_from_path (path: string): Mat.t =
  let buf = read_bin_file path in
  let count = Int32.to_int (Bytes.get_int32_be buf 4) in
  let rows = Int32.to_int (Bytes.get_int32_be buf 8) in
  let cols = Int32.to_int (Bytes.get_int32_be buf 12) in
  Mat.init count (rows * cols) (fun row col ->
      let pixel = Bytes.get_uint8 buf (16 + (row * rows * cols) + col) in
      (Float.of_int pixel) /. 255.)

let main () =
  print_endline "Read README for information on this model.";
  if Array.length Sys.argv < 2 then (
    Printf.printf "Supply the path to the directory containing uncompressed mnist databases.\n";
    Printf.printf "You can get them from 'http://yann.lecun.com/exdb/mnist/'.\n";
    Stdlib.exit 1
  );
  let dir_path = Sys.argv.(1) in

  let train_images = images_from_path (dir_path ^ "/train-images-idx3-ubyte") in
  Printf.printf "Read %ix%i training images.\n" train_images.rows train_images.cols;

  let train_labels = labels_from_path (dir_path ^ "/train-labels-idx1-ubyte") in
  Printf.printf "Read %ix%i training labels.\n" train_labels.rows train_labels.cols;

  let test_images = images_from_path (dir_path ^ "/t10k-images-idx3-ubyte") in
  Printf.printf "Read %ix%i testing images.\n" test_images.rows test_images.cols;

  let test_labels = labels_from_path (dir_path ^ "/t10k-labels-idx1-ubyte") in
  Printf.printf "Read %ix%i testing labels.\n" test_labels.rows test_labels.cols;

  let interrupted = ref false in
  Sys.set_signal Sys.sigint (Sys.Signal_handle (fun _ -> interrupted := true));

  Random.self_init ();
  let shape = [| 784; 128; 10; |] in
  let model = ref (Model.random_range shape 0.02 0.12) in

  let print_cost () =
    let (rand_tx, rand_ty) = Model.random_train_set test_images test_labels 2 in
    Printf.printf "Cost: %f\n" (Model.cost !model ~tx:rand_tx ~ty:rand_ty);
    Stdlib.flush Stdlib.stdout;
  in

  print_cost ();
  let iter = ref 0 in
  while !interrupted = false do
    iter := !iter + 1;

    let (rand_tx, rand_ty) = Model.random_train_set train_images train_labels 32 in
    let grad = Model.back_propagation !model ~tx:rand_tx ~ty:rand_ty ~rate:0.01 in
    model := Model.apply_gradient !model grad;

    if !iter mod 10 = 0 then
      print_cost ();
  done;
  print_endline "Interrupted\n";

  let (rand_tx, rand_ty) = Model.random_train_set test_images test_labels 10 in
  Model.evaluate !model ~tx:rand_tx ~ty:rand_ty;

;;
main ()
