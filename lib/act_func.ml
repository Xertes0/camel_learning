type t =
  { f : float -> float; (* Function itself *)
    d : float -> float; (* It's dervative.
                           The input is the RESULT from f,
                           not f's input! *)
  }

let sigmoid: t =
  { f = (fun x -> 1. /. (1. +. Float.exp (-. x)));
    d = (fun x -> x *. (1. -. x));
  }

(* This is a leaky ReLU.
   See "Potential problems" and "Variants":
   https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 *)
let relu: t =
  { f = (fun x -> if x > 0. then x else x *. 0.01);
    d = (fun x -> if x < 0. then 0.01 else 1.);
  }
