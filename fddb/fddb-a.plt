# Compare your discrete ROC curves with other methods
# At terminal: gnuplot discROC.p
set terminal png size 1280, 960 enhanced font 'Verdana,18'
set size 1,1
set grid
set title "FDDB-A Threshold (0.6, 0.7, 0.8)"
set ylabel "True positive rate"
set xlabel "False positive"
set key below
set output "fddb-a.png"
plot "fddb-a.txt" using 2:1 title 'jfda' with lines lw 2 , \
