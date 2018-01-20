<html>
<head>
<?php $y = $_COOKIE["y"];?>
<title>view</title>
<style>
img.positive {
    border: 10px solid red;
}
img.negative {
    border: 10px solid white;
}
img.ignored {
    opacity: 0.33;
    filter: alpha(opacity=33);
    border: 10px solid gray;
}
</style>
</head>
<body>

<?php
print "<body onScroll=\"document.cookie='y=' + window.pageYOffset\" onLoad='window.scrollTo(0,$y)'>";
?>

<?PHP
$page = $_GET['page'];
?>

<a href=view.php?page=<?PHP printf("%d", $page - 1);?>>Prev Page</a>
<a href=view.php?page=<?PHP printf("%d", $page + 1);?>>Next Page</a>
<p>
<?PHP printf('You are in page [<b>%d</b>]', $page); ?>
<p>
<?PHP
$n_per_page = 10;
for ($i = 1; $i <= $n_per_page; ++$i) {
    $im_id = $i + $page * $n_per_page;
    printf("<a href=\"click.php?im_id=%d\">\n", $im_id);
    if (file_exists(sprintf('clicks/clicked_%d.txt', $im_id))) {
        if (file_exists(sprintf('clicks/ignored_%d.txt', $im_id))) {
            printf("<img src=\"images/%d.png\" class=\"ignored\"/></a>\n\n", $im_id);
        } else {
            printf("<img src=\"images/%d.png\" class=\"positive\"/></a>\n\n", $im_id);
        }
    } else {
        printf("<img src=\"images/%d.png\" class=\"negative\"/></a>\n\n", $im_id);
    }
}
?>
<a href=view.php?page=<?PHP printf("%d", $page + 1);?>><font size="7">Next Page</font></a>

</body>
</html>

