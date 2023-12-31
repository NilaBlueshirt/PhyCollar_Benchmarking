#!/bin/sh
for file in /scratch/tianche5/PhyCollar/*/*/*/*/*auc.csv; 
do 
	parentdir="$(dirname "$file")";
	cat "$parentdir"/*_auc.csv > "$parentdir"/auc.csv;
	grep -v "Trial,Fold,AUC" "$parentdir"/auc.csv > temp && mv temp "$parentdir"/auc.csv;
done

for file in /scratch/tianche5/PhyCollar/*/*/*/*auc.csv; 
do 
	parentdir="$(dirname "$file")";
	cat "$parentdir"/*_auc.csv > "$parentdir"/auc.csv;
	grep -v "Trial,Fold,AUC" "$parentdir"/auc.csv > temp && mv temp "$parentdir"/auc.csv;
done

for file in /scratch/tianche5/PhyCollar/*/*/*/*/*pr.csv; 
do 
	parentdir="$(dirname "$file")";
	cat "$parentdir"/*_pr.csv > "$parentdir"/pr.csv;
	grep -v "Trial,Fold,PR-AUC" "$parentdir"/pr.csv > temp && mv temp "$parentdir"/pr.csv;
done

for file in /scratch/tianche5/PhyCollar/*/*/*/*pr.csv; 
do 
	parentdir="$(dirname "$file")";
	cat "$parentdir"/*_pr.csv > "$parentdir"/pr.csv;
	grep -v "Trial,Fold,PR-AUC" "$parentdir"/pr.csv > temp && mv temp "$parentdir"/pr.csv;
done

rm /scratch/tianche5/PhyCollar/*/slurm*
rm -r /scratch/tianche5/PhyCollar/*/catboost_info/