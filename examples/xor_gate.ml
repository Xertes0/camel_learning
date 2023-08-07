open Camel_learning

;;
Demo.run
  { tx = Matrix.from_3d_list
           [ [ 0.; 0. ];
             [ 0.; 1. ];
             [ 1.; 0. ];
             [ 1.; 1. ] ];
    ty = Matrix.from_3d_list
           [ [ 0. ];
             [ 1. ];
             [ 1. ];
             [ 0. ] ];
    shape = [| 2; 2; 1; |];
  } ()
