# Move everything to test set except some number of files
# $1 = Folder to select from
# $2 = Number of files to keep as training data
echo "Selecting $2 files from $1"
ls ./training/$1 | sort -R | tail -n+$(($2+1)) | while read file; do
	mkdir -p ./training/selected;
	mkdir -p ./training/selected/$1;
	mv -n ./training/$1/$file ./training/selected/$1/$file;
done