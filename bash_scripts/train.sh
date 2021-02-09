for seed in 0 1 2 3
do
python main.py -d wikipedia --pos_dim 0 --agg tree --bs 32 --n_degree 20 --n_layer 2 --mode i --seed $seed &
done
wait

for seed in 0 1 2 3
do
python main.py -d wikipedia --pos_dim 0 --agg tree --bs 32 --n_degree 20 --n_layer 2 --mode t --seed $seed &
done
wait

for seed in 0 1 2 3
do
python main.py -d reddit --pos_dim 0 --agg tree --bs 32 --n_degree 20 --n_layer 2 --mode i --seed $seed &
done
wait

for seed in 0 1 2 3
do
python main.py -d reddit --pos_dim 0 --agg tree --bs 32 --n_degree 20 --n_layer 2 --mode t --seed $seed &
done
wait

for seed in 0 1 2 3
do
python main.py -d socialevolve --pos_dim 0 --agg tree --bs 32 --n_degree 20 --n_layer 2 --mode i --seed $seed &
done
wait

for seed in 0 1 2 3
do
python main.py -d socialevolve --pos_dim 0 --agg tree --bs 32 --n_degree 20 --n_layer 2 --mode t --seed $seed &
done
wait


python main.py -d wikipedia --pos_dim 108 --agg walk --bs 32 --n_degree 100 1 1 --mode i

python main.py -d wikipedia --pos_dim 108 --agg walk --bs 32 --n_degree 100 1 1 --mode i