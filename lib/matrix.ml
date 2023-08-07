type t =
  { arr : float array;
    rows : int;
    cols : int;
  }

let zeros (rows: int) (cols: int): t =
  { arr = Array.make (rows * cols) 0.;
    rows = rows;
    cols = cols;
  }

let random_range (rows: int) (cols: int) (min: float) (max: float): t =
  { arr = Array.init (rows * cols) (fun _ -> (Random.float (max -. min) +. min));
    rows = rows;
    cols = cols;
  }

let random (rows: int) (cols: int): t =
  random_range rows cols 0. 1.

let from_3d_list (list: float list list): t =
  { arr = Array.of_list (List.flatten list);
    rows = List.length list;
    cols = List.length (List.hd list);
  }

let init (rows: int) (cols: int) (f: int -> int -> float): t =
  { arr = Array.init (rows * cols) (fun i -> f (i/cols) (i mod cols));
    rows = rows;
    cols = cols;
  }

let from_array (rows: int) (cols: int) (arr: float array): t =
  assert (Array.length arr = rows * cols);
  { arr = arr;
    rows = rows;
    cols = cols;
  }

let get (mat: t) x y =
  mat.arr.((x*mat.cols) + y)

let set (mat: t) x y value =
  mat.arr.((x*mat.cols) + y) <- value

let set_with (mat: t) (row: int) (col: int) (f: float -> float): unit =
  set mat row col (f (get mat row col))

let map (f: float -> float) (mat: t): t =
  { mat with arr = Array.map f mat.arr }

let show (mat: t): unit =
  print_string "[\n";
  for x = 0 to mat.rows - 1 do
    print_string "\t[";
    for y = 0 to mat.cols - 1 do
      Printf.printf "%f, " (get mat x y)
    done;
    print_string "]\n";
  done;
  print_string "]\n"

let sum (a: t) (b: t): t =
  assert (a.rows = b.rows &&
            a.cols = b.cols);
  { a with arr = Array.map2 ( +. ) a.arr b.arr }

let dot (a: t) (b: t): t =
  assert (a.cols = b.rows);
  let res = zeros a.rows b.cols in
  for x = 0 to res.rows - 1 do
    for y = 0 to res.cols - 1 do
      let sum = ref 0. in
      for z = 0 to a.cols - 1 do
        sum := !sum +. ((get a x z) *. (get b z y))
      done;
      res.arr.(x + (y * res.rows)) <- !sum;
    done
  done;
  res

let take_row (mat: t) (row: int): t =
  { arr = Array.sub mat.arr (row * mat.cols) mat.cols;
    rows = 1;
    cols = mat.cols;
  }
