division upwards : 

  int N = 1000000;
  int block_size = 512;
  int grid_size = (N + block_size - 1) / block_size;

  also remember donot start the extra thread!
