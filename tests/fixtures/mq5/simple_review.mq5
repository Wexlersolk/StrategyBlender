input double Lots = 1.5;
input int Period = 14;
input bool UseFilter = true;

int OnInit()
{
   Print("hello");
   return 0;
}

void OnTick()
{
   double x = Lots;
   if(Period > 10)
   {
      x = x + 1;
   }
   else
   {
      x = x - 1;
   }
}
