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

let get (mat: t) x y =
  mat.arr.((x*mat.cols) + y)

let set (mat: t) x y value =
  mat.arr.((x*mat.cols) + y) <- value

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
