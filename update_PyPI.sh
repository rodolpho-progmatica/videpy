echo '\n\n***********************************************\n'
echo 'Reminder: CHANGE THE VERSION IN setup.py'
echo '\n***********************************************\n\n'

rm dist/*

python setup.py sdist

twine upload dist/*

