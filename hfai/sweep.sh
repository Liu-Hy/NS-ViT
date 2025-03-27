for a in 1e-4 3e-4 3e-5
do
  for b in 3 2 1
  do
    for c in 0.1 0.03
    do
      for d in 0.01 0.1
      do
        python3 hfai/train_hfai.py  --lr $a  --lim $b  --nlr $c  --eps $d
      done
    done
  done
done
