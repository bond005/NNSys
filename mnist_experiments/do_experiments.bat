rem mnist_convertor.exe src_data prepared_data
rem copy prepared_data\mnist_trainset.dat prepared_data\mnist_trainset_full.dat
rem nnsys.exe -trainset=prepared_data\mnist_trainset.dat -controlset=prepared_data\mnist_controlset.dat -r=0.15

rem nnsys.exe -set=i784-10max -mlp=recognizers\untrained_mlp_i784_10max.dat -struct
rem nnsys.exe -mlp=recognizers\untrained_mlp_i784_10max.dat -init
rem nnsys.exe -set=i784-300sig-10max -mlp=recognizers\untrained_mlp_i784_300sig_10max.dat -struct
rem nnsys.exe -mlp=recognizers\untrained_mlp_i784_300sig_10max.dat -init
rem nnsys.exe -set=i784-300sig-100sig-10max -mlp=recognizers\untrained_mlp_i784_300sig_100sig_10max.dat -struct
rem nnsys.exe -mlp=recognizers\untrained_mlp_i784_300sig_100sig_10max.dat -init

rem copy recognizers\untrained_mlp_i784_10max.dat recognizers\mlp_i784_10max_v1.dat
rem nnsys.exe -mlp=recognizers\mlp_i784_10max_v1.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=bp_s -lr_init=1e-3 -lr_fin=1e-4 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_10max_v1.txt
rem nnsys.exe -mlp=recognizers\mlp_i784_10max_v1.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_10max_v1.txt

rem copy recognizers\untrained_mlp_i784_10max.dat recognizers\mlp_i784_10max_v2.dat
rem nnsys.exe -mlp=recognizers\mlp_i784_10max_v2.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=dbd -lr=5e-4 -theta=1e-4 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_10max_v2.txt
rem nnsys.exe -mlp=recognizers\mlp_i784_10max_v2.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_10max_v2.txt

copy recognizers\untrained_mlp_i784_10max.dat recognizers\mlp_i784_10max_v3.dat
nnsys.exe -mlp=recognizers\mlp_i784_10max_v3.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=bp_b -lr_max=5 -lr_iters=20 -eps=1e-6 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_10max_v3.txt
nnsys.exe -mlp=recognizers\mlp_i784_10max_v3.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_10max_v3.txt

copy recognizers\untrained_mlp_i784_10max.dat recognizers\mlp_i784_10max_v4.dat
nnsys.exe -mlp=recognizers\mlp_i784_10max_v4.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=cg -lr_max=5 -lr_iters=20 -eps=1e-6 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_10max_v4.txt
nnsys.exe -mlp=recognizers\mlp_i784_10max_v4.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_10max_v4.txt

copy recognizers\untrained_mlp_i784_10max.dat recognizers\mlp_i784_10max_v5.dat
nnsys.exe -mlp=recognizers\mlp_i784_10max_v5.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=rp -lr=1e-1 -eps=1e-6 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_10max_v5.txt
nnsys.exe -mlp=recognizers\mlp_i784_10max_v5.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_10max_v5.txt

copy recognizers\untrained_mlp_i784_300sig_10max.dat recognizers\mlp_i784_300sig_10max_v1.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v1.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=bp_s -lr_init=1e-3 -lr_fin=1e-4 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_10max_v1.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v1.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_10max_v1.txt

copy recognizers\untrained_mlp_i784_300sig_10max.dat recognizers\mlp_i784_300sig_10max_v2.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v2.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=dbd -lr=5e-4 -theta=1e-4 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_10max_v2.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v2.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_10max_v2.txt

copy recognizers\untrained_mlp_i784_300sig_10max.dat recognizers\mlp_i784_300sig_10max_v3.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v3.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=bp_b -lr_max=5 -lr_iters=20 -eps=1e-6 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_10max_v3.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v3.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_10max_v3.txt

copy recognizers\untrained_mlp_i784_300sig_10max.dat recognizers\mlp_i784_300sig_10max_v4.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v4.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=cg -lr_max=5 -lr_iters=20 -eps=1e-6 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_10max_v4.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v4.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_10max_v4.txt

copy recognizers\untrained_mlp_i784_300sig_10max.dat recognizers\mlp_i784_300sig_10max_v5.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v5.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=rp -lr=1e-1 -eps=1e-6 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_10max_v5.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_10max_v5.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_10max_v5.txt

copy recognizers\untrained_mlp_i784_300sig_100sig_10max.dat recognizers\mlp_i784_300sig_100sig_10max_v1.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v1.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=bp_s -lr_init=1e-3 -lr_fin=1e-4 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_100sig_10max_v1.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v1.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_100sig_10max_v1.txt

copy recognizers\untrained_mlp_i784_300sig_100sig_10max.dat recognizers\mlp_i784_300sig_100sig_10max_v2.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v2.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=dbd -lr=5e-4 -theta=1e-4 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_100sig_10max_v2.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v2.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_100sig_10max_v2.txt

copy recognizers\untrained_mlp_i784_300sig_100sig_10max.dat recognizers\mlp_i784_300sig_10max_100sig_v3.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v3.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=bp_b -lr_max=5 -lr_iters=20 -eps=1e-6 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_100sig_10max_v3.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v3.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_100sig_10max_v3.txt

copy recognizers\untrained_mlp_i784_300sig_100sig_10max.dat recognizers\mlp_i784_300sig_100sig_10max_v4.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v4.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=cg -lr_max=5 -lr_iters=20 -eps=1e-6 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_100sig_10max_v4.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v4.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_100sig_10max_v4.txt

copy recognizers\untrained_mlp_i784_300sig_100sig_10max.dat recognizers\mlp_i784_300sig_100sig_10max_v5.dat
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v5.dat -train=prepared_data\mnist_trainset.dat -control=prepared_data\mnist_controlset.dat -alg=rp -lr=1e-1 -eps=1e-6 -esmooth=13 -earlystop -maxepochs=500 -restarts=3 -etr -eg -log > log\trainlog_of_mlp_i784_300sig_100sig_10max_v5.txt
nnsys.exe -mlp=recognizers\mlp_i784_300sig_100sig_10max_v5.dat -in=prepared_data\mnist_testset.dat -task=class > log\testlog_of_mlp_i784_300sig_100sig_10max_v5.txt
