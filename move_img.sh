for name in `ls data/p0.025s2__03-04-2023_03-10-35/img`; 
do 
mv data/p0.025s2__03-04-2023_03-10-35/img/$name img/p0.025a/${name%.jpg}_1.jpg;
done