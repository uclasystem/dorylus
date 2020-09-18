// Remove the header of feature files for DGL loading.
// Take reddit-small as an example.
// Input:  reddit-feat.bin
// Output: new-feat.bin

#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdlib.h>

int main()
{
    struct stat statbuf;
    char asin[11];
    int cnt = 0;

    char in_file[20];
    char out_file[20];
    sprintf(in_file, "reddit-feat.bin");
    sprintf(out_file, "new-feat.bin");
    if (stat(in_file, &statbuf) != -1)
    {
        FILE *fd = fopen(in_file, "rb");
        FILE *outfd = fopen(out_file, "wb");

        int svt = 4;
        unsigned featdim = 602;
        unsigned numVtcs = 232965;

        unsigned *buf = malloc(sizeof(float) * featdim * numVtcs);

        fwrite(&featdim, sizeof(unsigned), 1, outfd);

        {
            if (fread(buf, sizeof(float), featdim * numVtcs, fd) != featdim * numVtcs)
            {
                printf("ERROR!\n");
            }
            fwrite(buf, sizeof(float), featdim * numVtcs, outfd);
        }

        fclose(fd);
        fclose(outfd);

        free(buf);
    }
    return 0;
}
