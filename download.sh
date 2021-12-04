#!/usr/bin/env sh

mysql CORA --host=relational.fit.cvut.cz --user=guest --password=relational -e "select * from paper" | sed "s/\t/,/g" >paper.csv
mysql CORA --host=relational.fit.cvut.cz --user=guest --password=relational -e "select * from cites" | sed "s/\t/,/g" >cites.csv
mysql CORA --host=relational.fit.cvut.cz --user=guest --password=relational -e "select * from content" | sed "s/\t/,/g" >content.csv
