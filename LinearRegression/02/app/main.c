#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


typedef struct
{
    float a;
    float b;
}LinReg;


#define lin_reg_init (LinReg)                   \
    {                                           \
        .a=(float) rand() / (float)RAND_MAX     \
        , .b=(float) rand()/(float)RAND_MAX     \
    }


/*
** error computation
*/
float
error(float y, float y_p)
{
    return powf((y - y_p), 2);
}


/*
** model evaluation based on input value 'x'
*/
float
forward(LinReg * self, float x)
{
    return (x * self->a) + self->b;
}


/*
** Gradient for 'a' parameter is simply partial derivative regarding to parameter a
** This is information about impact of change of value of parameter 'a' to change of output loss
*/
float
gradient_a(
    LinReg * self
    , float x
    , float y)
{
    return -2 * x * (y - forward(self, x));
}


/*
** Gradient for 'b' parameter is simply partial derivative regarding to parameter a
** This is information about impact of change of value of parameter 'b' to change of output loss
*/
float
gradient_b(
    LinReg * self
    , float x
    , float y)
{
    return -2 * (y - forward(self, x));
}


int
main(void)
{
    srand(time(NULL));

    /*
    ** traing dataset
    */
    float X[] = {1, 2, 3, 4}; 
    float Y[] = {3, 5, 7, 9}; // (x*2 + 1)

    /*
    ** initialization of linear regression model with random values
    */
    LinReg model = lin_reg_init;

    size_t epochs     = 500;
    size_t batch_size = sizeof(X)/sizeof(*X);
    float lr          = 0.1;
    
    for(size_t epoch = 0; epoch < epochs; epoch++)
    {
        float da   = 0;
        float db   = 0;
        float loss = 0;
        
        /*
        ** batch processing
        */
        for(size_t index = 0; index < batch_size; index++)
        {
            float y_p      = forward(&model, X[index]);
            float loc_loss = error(Y[index], y_p); 
            loss          += loc_loss;
            da            += gradient_a(&model, X[index], Y[index]);
            db            += gradient_b(&model, X[index], Y[index]);
        }

        loss /= batch_size;

        /*
        ** parameters update 
        */ 
        model.a -= da / batch_size * lr;
        model.b -= db / batch_size * lr;

        printf(
            "epoch: %ld, loss: %f - {a: %f, b: %f} {da: %f, db: %f}\n"
            , epoch+1
            , loss
            , model.a, model.b
            , da, db);

        if(loss <= 0.00000000001)
            break;
    }

    printf("Program exit..\n");

    return EXIT_SUCCESS;
}



