#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


typedef struct
{
    float a;
    float b;
}LinReg;


#define lin_reg (LinReg)                        \
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


#define DELTA 0.000001


/*
** computation of gradient for 'a' parameter - partial derivative with respect to parametr 'a'
** computation is based on finite difference: dt = (f(x+delta) - f(x)) / delta
** which is numerical derivation in essence
*/
float
gradient_a(
    LinReg * self
    , float x
    , float y
    , float loss)
{
    float delta_loss = powf(y- ((x * (self->a + DELTA)) + self->b), 2);

    return (delta_loss - loss) / DELTA;
}


/*
** computation of gradient for 'b' parameter - partial derivative with respect to parametr 'b'
** computation is based on finite difference: dt = (f(x+delta) - f(x)) / delta
** which is numerical derivation in essence
*/
float
gradient_b(
    LinReg * self
    , float x
    , float y
    , float loss)
{
    float delta_loss = powf(y- ((x * self->a) + (self->b+DELTA)), 2);

    return (delta_loss - loss) / DELTA;
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
    LinReg model = lin_reg;

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
            da            += gradient_a(&model, X[index], Y[index], loc_loss);
            db            += gradient_b(&model, X[index], Y[index], loc_loss);
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

        if(loss <= 0.00000001)
            break;
    }

    printf("Program exit..\n");

    return EXIT_SUCCESS;
}



