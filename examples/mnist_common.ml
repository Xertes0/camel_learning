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

type mnist =
  { tx : Mat.t; (* Training samples *)
    ty : Mat.t;
    ver_x : Mat.t; (* Verification samples *)
    ver_y : Mat.t;
  }

let mnist_from_path (dir_path: string): mnist =
  { tx = images_from_path (dir_path ^ "/train-images-idx3-ubyte");
    ty = labels_from_path (dir_path ^ "/train-labels-idx1-ubyte");
    ver_x = images_from_path (dir_path ^ "/t10k-images-idx3-ubyte");
    ver_y = labels_from_path (dir_path ^ "/t10k-labels-idx1-ubyte");
  }
