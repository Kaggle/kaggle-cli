#!/bin/bash

# Verify all options are plumbed through to the MT.
# Set envar KAGGLE_DEVELOPER to the Kaggle user name (probably already done).
# When prompted to delete something, respond with "no".

# Still need to adjust for assumptions about existing artifacts, like
# the notebook "exercise-as-with"

kaggle --version

echo "kaggle competitions files"
kaggle competitions files titanic --page-size=3 --page-token=abcd -v -q
echo "kaggle competitions list"
kaggle competitions list --group general --category featured --sort-by prize
echo "kaggle competitions download"
kaggle c download titanic -w -o -q
kaggle c download titanic -f test.csv -p tost
echo "kaggle competitions submit"
kaggle c download house-prices-advanced-regression-techniques -f sample_submission.csv --force
kaggle c submit house-prices-advanced-regression-techniques -f sample_submission.csv -m "Test message"
echo "kaggle competitions submissions"
kaggle c submissions house-prices-advanced-regression-techniques -v -q
echo "kaggle competitions leaderboard"
kaggle c leaderboard titanic -v -q -d -p leaders
kaggle c leaderboard titanic -s > leaderboard.txt
echo "kaggle competitions pages"
kaggle c pages titanic
kaggle c pages titanic -v -q
kaggle c pages titanic --page-name rules --content
rm -r titanic.zip tost sample_submission.csv leaders leaderboard.txt

echo "kaggle kernels list"
kaggle k list -m -s Exercise --page-size 5 -p 2 -v  --sort-by dateRun
kaggle k list --parent $KAGGLE_DEVELOPER/exercise-lists
kaggle k list --competition house-prices-advanced-regression-techniques --page-size 5
kaggle k list --dataset dansbecker/home-data-for-ml-course --page-size 5
kaggle k list --user $KAGGLE_DEVELOPER --language python --kernel-type notebook --output-type data
echo "kaggle kernels files"
kaggle kernels files kerneler/sqlite-global-default -v --page-size=1
echo "kaggle kernels init"
kaggle k init -p tests/kernel
echo "kaggle kernels get"
kaggle k get -p tests/kernel $KAGGLE_DEVELOPER/exercise-as-with -m
kaggle k get --wp $KAGGLE_DEVELOPER/exercise-as-with
#sed -i s/exercise-as-with/exercise-delete/ tests/kernel/kernel-metadata.json
#sed -i s/As \& With/Delete/ tests/kernel/kernel-metadata.json
#mv tests/kernel/exercise-as-with.ipynb tests/kernel/exercise-delete.ipynb
echo "kaggle kernels update"
kaggle kernels update -p tests/kernel
echo "kaggle kernels status"
kaggle k status kerneler/sqlite-global-default
echo "kaggle kernels output"
kaggle k output kerneler/sqlite-global-default -o
echo "kaggle kernels delete"
#kaggle k delete $KAGGLE_DEVELOPER/exercise-delete
#kaggle k delete $KAGGLE_DEVELOPER/exercise-delete --yes
rm -f tests/kernel/exercise-as-with.ipynb tests/kernel/kernel-metadata.json exercise-as-with.ipynb

echo "kaggle datasets list"
kaggle d list --size 10
kaggle d list -m
kaggle d list --user oktayrdeki --csv
kaggle d list --file-type csv --page 2 --sort-by updated -s student --min-size 13000 --max-size 15000
kaggle d list --license odb --tags internet --search telco
echo "kaggle datasets files"
kaggle datasets files kerneler/brazilian-bird-observation-metadata-from-wikiaves --page-size=7 --page-token=abcd
echo "kaggle datasets init"
kaggle d init -p tests/dataset
echo "kaggle datasets create"
export SLUG=testing
sed -i s/INSERT_TITLE_HERE/TitleHere/ tests/dataset/dataset-metadata.json
sed -i s/INSERT_SLUG_HERE/$SLUG/ tests/dataset/dataset-metadata.json
kaggle d create -p tests/dataset --public -q -t -r skip
echo "kaggle datasets download"
kaggle datasets download -d willianoliveiragibin/pixar-films
kaggle d download goefft/public-datasets-with-file-types-and-columns -p tmp --unzip -o -q
kaggle d download goefft/public-datasets-with-file-types-and-columns -f dataset_results.csv -w -q -o
echo "kaggle datasets version"
kaggle d version -m VersionNotesGoHere -p tests/dataset -q -t -r skip -d
echo "kaggle datasets metadata"
kaggle datasets metadata goefft/public-datasets-with-file-types-and-columns -p tests/dataset
echo "kaggle datasets status"
kaggle d status goefft/public-datasets-with-file-types-and-columns
echo "kaggle datasets delete"
kaggle d delete $KAGGLE_DEVELOPER/$SLUG
kaggle d delete $KAGGLE_DEVELOPER/$SLUG --yes
rm -rf tmp tests/dataset/dataset-metadata.json dataset_results.csv.zip dataset_results.csv pixar-films.zip

echo "kaggle models init"
mkdir tmp
kaggle m init -p tmp
echo "kaggle models list"
kaggle m list --owner $KAGGLE_DEVELOPER --sort-by createTime -v
kaggle m list -s gemini --page-size 5
echo "kaggle models create"
sed -i s/INSERT_OWNER_SLUG_HERE/$KAGGLE_DEVELOPER/ tmp/model-metadata.json
sed -i s/INSERT_TITLE_HERE/ModelTitle/ tmp/model-metadata.json
sed -i s/INSERT_SLUG_HERE/test-model/ tmp/model-metadata.json
kaggle m create -p tmp
echo "kaggle models update"
kaggle m update -p tmp
sleep 10
echo "kaggle models get"
kaggle m get -p tmp $KAGGLE_DEVELOPER/test-model

echo "kaggle models instances init"
kaggle m instances init -p tmp
echo "kaggle models instances create"
sed -i s/INSERT_OWNER_SLUG_HERE/$KAGGLE_DEVELOPER/ tmp/model-instance-metadata.json
sed -i s/INSERT_EXISTING_MODEL_SLUG_HERE/test-model/ tmp/model-instance-metadata.json
sed -i s/INSERT_INSTANCE_SLUG_HERE/main/ tmp/model-instance-metadata.json
sed -i s/INSERT_FRAMEWORK_HERE/jax/ tmp/model-instance-metadata.json
echo "a,b,c,d" > tmp/data.csv
kaggle models instances create -p tmp -q -r skip
sleep 10
echo "kaggle models instances update"
kaggle models instances update -p tmp
sleep 10
echo "kaggle models instances get"
kaggle models instances get $KAGGLE_DEVELOPER/test-model/jax/main -p tmp
echo "kaggle models instances files"
kaggle models instances files $KAGGLE_DEVELOPER/test-model/jax/main -v --page-size 5

echo "kaggle models instances versions files"
kaggle models instances versions files google/gemma/pytorch/7b/2 -v --page-size=3 --page-token=abcd
echo "kaggle models instances versions create"
kaggle models instances versions create -p tmp -q -r skip -n VersionNotes $KAGGLE_DEVELOPER/test-model/jax/main
echo "kaggle models instances versions download"
kaggle models instances versions download -p tmp -q -f --untar $KAGGLE_DEVELOPER/test-model/jax/main/1

rm -rf tmp
rm -f results.csv

echo "kaggle models instances versions delete"
kaggle m instances versions delete $KAGGLE_DEVELOPER/test-model/jax/main/1
kaggle m instances versions delete $KAGGLE_DEVELOPER/test-model/jax/main/1 -y
echo "kaggle models instances delete"
kaggle m instances delete $KAGGLE_DEVELOPER/test-model/jax/main
kaggle m instances delete $KAGGLE_DEVELOPER/test-model/jax/main -y
echo "kaggle models delete"
kaggle m delete $KAGGLE_DEVELOPER/test-model
kaggle m delete $KAGGLE_DEVELOPER/test-model -y
