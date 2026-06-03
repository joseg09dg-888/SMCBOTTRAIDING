//+------------------------------------------------------------------+
//|  SMCBotEA.mq5 — Reads Python bot signals from MT5 Common\Files  |
//|  Compile in MetaEditor (F7), attach to EURUSD chart              |
//|  Python writes to: AppData\MetaQuotes\Terminal\Common\Files\     |
//+------------------------------------------------------------------+
#property copyright "SMC Trading Bot"
#property version   "1.10"
#property strict

// File names — stored in Common\Files (accessible by Python and all MT5)
#define SIGNAL_FILE "smc_signal.json"
#define RESULT_FILE "smc_result.json"
#define LOCK_FILE   "smc_lock.tmp"

//+------------------------------------------------------------------+
int OnInit()
{
   EventSetTimer(3);
   Print("SMC Bot EA v1.10 started — watching Common\\Files\\", SIGNAL_FILE);
   return(INIT_SUCCEEDED);
}
void OnDeinit(const int reason) { EventKillTimer(); }
void OnTick() {}

//+------------------------------------------------------------------+
void OnTimer()
{
   if(FileIsExist(LOCK_FILE, FILE_COMMON)) return;
   if(!FileIsExist(SIGNAL_FILE, FILE_COMMON)) return;

   // Lock
   int lh = FileOpen(LOCK_FILE, FILE_WRITE|FILE_TXT|FILE_COMMON|FILE_ANSI);
   if(lh == INVALID_HANDLE) return;
   FileWriteString(lh, "1"); FileClose(lh);

   // Read signal
   int fh = FileOpen(SIGNAL_FILE, FILE_READ|FILE_TXT|FILE_COMMON|FILE_ANSI);
   if(fh == INVALID_HANDLE) { FileDelete(LOCK_FILE,FILE_COMMON); return; }
   string json="";
   while(!FileIsEnding(fh)) json += FileReadString(fh);
   FileClose(fh);
   FileDelete(SIGNAL_FILE, FILE_COMMON);
   Print("Signal: ", json);

   // Parse
   string symbol  = ParseStr(json, "symbol",    "EURUSD");
   string dir     = ParseStr(json, "direction", "long");
   double volume  = ParseDbl(json, "volume",    0.01);
   double sl_pips = ParseDbl(json, "sl_pips",   50.0);
   double tp_pips = ParseDbl(json, "tp_pips",   100.0);

   ENUM_ORDER_TYPE otype = (dir=="long"||dir=="LONG") ?
                            ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   if(!SymbolSelect(symbol, true)) {
      WriteResult("{\"retcode\":4301,\"error\":\"symbol not found\",\"order\":0}");
      FileDelete(LOCK_FILE, FILE_COMMON); return;
   }

   double pt  = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int    dg  = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double price, sl, tp;
   if(otype == ORDER_TYPE_BUY) {
      price = SymbolInfoDouble(symbol, SYMBOL_ASK);
      sl = NormalizeDouble(price - sl_pips*pt*10, dg);
      tp = NormalizeDouble(price + tp_pips*pt*10, dg);
   } else {
      price = SymbolInfoDouble(symbol, SYMBOL_BID);
      sl = NormalizeDouble(price + sl_pips*pt*10, dg);
      tp = NormalizeDouble(price - tp_pips*pt*10, dg);
   }

   MqlTradeRequest req={}; MqlTradeResult res={};
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = symbol;
   req.volume    = volume;
   req.type      = otype;
   req.price     = price;
   req.sl        = sl;
   req.tp        = tp;
   req.deviation = 20;
   req.magic     = 234000;
   req.comment   = "SMC Bot";
   req.type_time = ORDER_TIME_GTC;
   req.type_filling = ORDER_FILLING_IOC;

   bool ok = OrderSend(req, res);
   string result = StringFormat(
      "{\"retcode\":%d,\"order\":%d,\"price\":%.5f,"
      "\"sl\":%.5f,\"tp\":%.5f,\"symbol\":\"%s\",\"ok\":%s}",
      res.retcode, res.order, res.price, sl, tp,
      symbol, ok?"true":"false"
   );
   WriteResult(result);
   FileDelete(LOCK_FILE, FILE_COMMON);

   if(res.retcode == TRADE_RETCODE_DONE)
      Print("Trade OK! Order=",res.order," Price=",res.price);
   else
      Print("Trade failed retcode=",res.retcode);
}

void WriteResult(string j)
{
   int h = FileOpen(RESULT_FILE, FILE_WRITE|FILE_TXT|FILE_COMMON|FILE_ANSI);
   if(h!=INVALID_HANDLE){ FileWriteString(h,j); FileClose(h); }
}

string ParseStr(string j, string k, string def)
{
   string s="\""+k+"\":\""; int i=StringFind(j,s);
   if(i<0) return def; i+=StringLen(s);
   int e=StringFind(j,"\"",i); if(e<0) return def;
   return StringSubstr(j,i,e-i);
}

double ParseDbl(string j, string k, double def)
{
   string s="\""+k+"\":"; int i=StringFind(j,s);
   if(i<0) return def; i+=StringLen(s);
   int e=i;
   while(e<StringLen(j)){ushort c=StringGetCharacter(j,e);if(c==','||c=='}') break; e++;}
   return StringToDouble(StringSubstr(j,i,e-i));
}
//+------------------------------------------------------------------+
