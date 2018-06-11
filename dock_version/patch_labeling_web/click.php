<html>
<head>
<title>click</title>
</head>
<body>

<?PHP
$im_id = $_GET['im_id'];

if (file_exists(sprintf('clicks/clicked_%d.txt', $im_id))) {
    if (file_exists(sprintf('clicks/ignored_%d.txt', $im_id))) {
        unlink(sprintf('clicks/clicked_%d.txt', $im_id));
        unlink(sprintf('clicks/ignored_%d.txt', $im_id));
    } else {
        touch(sprintf('clicks/ignored_%d.txt', $im_id));
    }
} else {
    touch(sprintf('clicks/clicked_%d.txt', $im_id));
}

header('Location: ' . $_SERVER['HTTP_REFERER']);
?>

</body>
</html>


