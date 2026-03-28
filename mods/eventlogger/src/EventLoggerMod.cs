using Vintagestory.API.Common;
using Vintagestory.API.Common.Entities;
using Vintagestory.API.Server;

namespace EventLogger
{
    public class EventLoggerMod : ModSystem
    {
        private ICoreServerAPI sapi = null!;

        public override bool ShouldLoad(EnumAppSide forSide) => forSide == EnumAppSide.Server;

        public override void StartServerSide(ICoreServerAPI api)
        {
            sapi = api;
            api.Event.PlayerDeath += OnPlayerDeath;
            api.Event.OnEntityDeath += OnEntityDeath;
        }

        private void OnPlayerDeath(IServerPlayer player, DamageSource? dmg)
        {
            string cause = "unknown";
            if (dmg != null)
            {
                Entity? causeEnt = dmg.CauseEntity ?? dmg.SourceEntity;
                cause = causeEnt?.Code?.Path ?? dmg.Type.ToString();
            }
            sapi.Logger.Event($"[WATCHDOG] PLAYER_DEATH {player.PlayerName} by {cause}");
        }

        private void OnEntityDeath(Entity entity, DamageSource? dmg)
        {
            if (dmg == null) return;
            if (entity is EntityPlayer) return; // obsługiwane przez PlayerDeath

            Entity? causeEnt = dmg.CauseEntity ?? dmg.SourceEntity;
            if (causeEnt is not EntityPlayer ep) return;

            IServerPlayer? killer = ep.Player as IServerPlayer;
            if (killer == null) return;

            string entityType = entity.Code?.Path ?? "unknown";
            sapi.Logger.Event($"[WATCHDOG] ENTITY_KILL {killer.PlayerName} killed {entityType}");
        }
    }
}
