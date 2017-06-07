import os

def main():

	#RNN Sizes
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_RNN100 --dropout 0 --rnn_size 100 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D0_RNN100 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_RNN100.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_RNN150 --dropout 0 --rnn_size 150 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D0_RNN150 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_RNN150.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_RNN200 --dropout 0 --rnn_size 200 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D0_RNN200 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_RNN200.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_RNN250 --dropout 0 --rnn_size 250 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D0_RNN250 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_RNN250.prb --calc --beam 1")

	#Highway Size
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_HW2 --dropout 0 --highway_layers 2 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D0_HW2 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_HW2.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_HW3 --dropout 0 --highway_layers 3 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D0_HW3 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_HW3.prb --calc --beam 1")

	#Go over some dropout values D = 0.1, 0.3, 0.5, 0.7 with RNN size = 150
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch50 --dropout 0 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D0_batch50 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch50.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D01_batch50 --dropout 0.1 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D01_batch50 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D01_batch50.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D03_batch50 --dropout 0.3 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D03_batch50 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D03_batch50.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D05_batch50 --dropout 0.5 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D05_batch50 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D05_batch50.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D07_batch50 --dropout 0.7 --batch_size 50")
	os.system("python evaluate.py --model cv/baseline_D07_batch50 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D07_batch50.prb --calc --beam 1")

	#Delay
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch50_d2 --dropout 0 --batch_size 50 --delay 2")
	os.system("python evaluate.py --model cv/baseline_D0_batch50_d2 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch50_d2.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch50_d3 --dropout 0.3 --batch_size 50 --delay 3")
	os.system("python evaluate.py --model cv/baseline_D0_batch50_d3 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch50_d3.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch50_d4 --dropout 0.5 --batch_size 50 --delay 4")
	os.system("python evaluate.py --model cv/baseline_D0_batch50_d4 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch50_d4.prb --calc --beam 1")

	#Word embedding
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch50_WE --dropout 0 --batch_size 50 --use_words 1 --use_chars 0")
	os.system("python evaluate.py --model cv/baseline_D0_batch50_WE --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch50_WE.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch50_WE_D03 --dropout 0.3 --batch_size 50 --use_words 1 --use_chars 0")
	os.system("python evaluate.py --model cv/baseline_D0_batch50_WE_D03 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch50_WE_D03.prb --calc --beam 1")

	#Batch Size
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch30 --dropout 0 --batch_size 30")
	os.system("python evaluate.py --model cv/baseline_D0_batch30 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch30.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch40 --dropout 0 --batch_size 40")
	os.system("python evaluate.py --model cv/baseline_D0_batch40 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch40.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch60 --dropout 0 --batch_size 60")
	os.system("python evaluate.py --model cv/baseline_D0_batch60 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch60.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch70 --dropout 0 --batch_size 70")
	os.system("python evaluate.py --model cv/baseline_D0_batch70 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch70.prb --calc --beam 1")
	os.system("python train.py --data_dir data/TXT --vector_size 0 --use_prb 0 --savefile baseline_D0_batch80 --dropout 0 --batch_size 80")
	os.system("python evaluate.py --model cv/baseline_D0_batch80 --vocabulary data/TXT/vocab.npz  --init init.npy --itext data/TXT/test.ctm  --otext baseline_D0_batch80.prb --calc --beam 1")

if __name__=="__main__":
	main()