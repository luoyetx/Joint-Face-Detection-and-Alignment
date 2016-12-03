# Compare your discrete ROC curves with other methods
# At terminal: gnuplot discROC.p
set terminal png size 1280, 960 enhanced font 'Verdana,18'
set size 1,1
set grid
set title "ONet-threshold-0.4"
set ylabel "True positive rate"
set xlabel "False positive"
set key below
set output "fddb-onet.png"
plot "caffe-mtcnn-onet-24DiscROC.txt" using 2:1 title 'caffe-mtcnn' with lines lw 2 , \
"mx-mtcnn-onet24-DiscROC.txt" using 2:1 title 'mx-mtcnn' with lines lw 2 , \
"jfda-onet-24DiscROC.txt" using 2:1 title 'jfda' with lines lw 2 , \
